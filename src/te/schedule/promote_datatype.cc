/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file promote_datatype.cc
 */
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr_functor.h>

#include <stack>
#include <map>

#include "graph.h"
#include "../../tir/transforms/ir_util.h"

/* Operations:
 * - PlaceholderOpNode:
 *   do nothing
 *   可能会出点问题，这里（可能不会，把 placeholder 的 GetShape 去掉之后 test_narrow_type 仍然没问题）
 * - ComputeOpNode
 *  - Update Var
 *  - Use the updated Var in body
 *  - Replace Inputs
 * - ScanOpNode
 *   - Dependent on its update node (ComputeOpNode)
 *   - May need to replace its scan_axis
 *   - 在哪里我们会把 Update inline 到 ScanOpNode 里面来？ScheduleOps?
 *   - scan_axis 同 update 里的第一维变量是同一个变量吗
 * - ExternOpNode:
 *   do nothing
 * - TensorComputeOpNode
 *   do nothing
 * - HybridOpNode
 *   do nothing
 * */

namespace tvm {
namespace te {

using namespace tir;

DataType GetPromotedDataType(Array<IterVar> vars) {
  int bits = 64;
  for (const auto& v : vars) {
    PrimExpr min = v->dom->min;
    PrimExpr extent = v->dom->extent;
    if (!min.as<IntImmNode>()) {
      bits = std::min(min->dtype.bits(), bits);
    }
    if (!extent.as<IntImmNode>()) {
      bits = std::min(extent->dtype.bits(), bits);
    }
  }
  return DataType::Int(bits);
}

IterVar cast(DataType dtype, IterVar iv) {
  Var v = Var(iv->var->name_hint, dtype);
  Range dom = Range(cast(dtype, iv->dom->min), cast(dtype, iv->dom->extent));
  return IterVar(dom, v, iv->iter_type, iv->thread_tag);
}

Array<IterVar> cast(DataType dtype, Array<IterVar> ivs) {
  std::vector<IterVar> ret;
  ret.reserve(ivs.size());
  for (const auto& iv : ivs) {
    ret.push_back(cast(dtype, iv));
  }
  return Array<IterVar>(ret.begin(), ret.end());
}

class DataTypeRewriter : public ExprMutator {
 public:
  DataTypeRewriter(Map<Var, Var> vmap, DataType target_dtype): 
    vmap_(vmap), target_dtype_(target_dtype) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    if (vmap_.find(var) == vmap_.end()) {
      return var;
    }
    return vmap_[var];
  }

  PrimExpr VisitExpr_(const ReduceNode* op) final {
    std::vector<IterVar> ivs;
    ivs.reserve(op->axis.size());
    for (const auto& iv : op->axis) {
      IterVar new_iv = cast(target_dtype_, iv);
      vmap_.Set(iv->var, new_iv->var);
      ivs.push_back(new_iv);
    }
    Array<IterVar> new_axis(ivs.begin(), ivs.end());
    Reduce new_op = Reduce(op->combiner, op->source, new_axis, op->condition, op->value_index);
    return  ExprMutator::VisitExpr_(new_op.as<ReduceNode>());
  }

  PrimExpr VisitExpr_(const AddNode* op) final;
  PrimExpr VisitExpr_(const SubNode* op) final;
  PrimExpr VisitExpr_(const MulNode* op) final;
  PrimExpr VisitExpr_(const DivNode* op) final;
  PrimExpr VisitExpr_(const ModNode* op) final;
  PrimExpr VisitExpr_(const FloorDivNode* op) final;
  PrimExpr VisitExpr_(const FloorModNode* op) final;
  PrimExpr VisitExpr_(const MinNode* op) final;
  PrimExpr VisitExpr_(const MaxNode* op) final;
  PrimExpr VisitExpr_(const EQNode* op) final;
  PrimExpr VisitExpr_(const NENode* op) final;
  PrimExpr VisitExpr_(const LTNode* op) final;
  PrimExpr VisitExpr_(const LENode* op) final;
  PrimExpr VisitExpr_(const GTNode* op) final;
  PrimExpr VisitExpr_(const GENode* op) final;
  PrimExpr VisitExpr_(const CallNode* op) final;
 private:
  Map<Var, Var> vmap_;
  DataType target_dtype_;
};

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC) \
  PrimExpr DataTypeRewriter::VisitExpr_(const OP* op) {   \
    PrimExpr a = this->VisitExpr(op->a);                  \
    PrimExpr b = this->VisitExpr(op->b);                  \
    if (a.same_as(op->a) && b.same_as(op->b)) {           \
      return GetRef<PrimExpr>(op);                        \
    } else {                                              \
      return FUNC(a, b);                                  \
    }                                                     \
  }

DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(ModNode, truncmod);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorDivNode, floordiv);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorModNode, floormod);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(EQNode, operator==);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(NENode, operator!=);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator<);  // NOLINT(*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator>);  // NOLINT(*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=);

PrimExpr DataTypeRewriter::VisitExpr_(const CallNode* op) {
  PrimExpr e = ExprMutator::VisitExpr_(op);
  op = e.as<CallNode>();
  CHECK(op != nullptr) << "Expected type to be CallNode"
                       << ", but get " << e->GetTypeKey();
  if (op->call_type == CallNode::PureIntrinsic) {
    if (op->name == intrinsic::tvm_if_then_else) {
      return if_then_else(op->args[0], op->args[1], op->args[2]);
    } else if (op->name == CallNode::shift_right) {
      return op->args[0] >> op->args[1];
    } else if (op->name == CallNode::shift_left) {
      return op->args[0] << op->args[1];
    } else if (op->name == CallNode::bitwise_and) {
      return op->args[0] & op->args[1];
    } else if (op->name == CallNode::bitwise_or) {
      return op->args[0] | op->args[1];
    } else if (op->name == CallNode::bitwise_xor) {
      return op->args[0] ^ op->args[1];
    } else if (op->name == "pow") {
      return pow(op->args[0], op->args[1]);
    }
  }
  return e;
}

Map<Operation, Operation> PromoteDataType(Array<Operation> ops) {
  // sort the ops in topological order
  ReadGraph graph = CreateReadGraph(ops);
  ops = PostDFSOrder(ops, graph);
  // iterate over the ops: promote IterVar datatype (need a pass, and a rule), replace inputs
  Map<Operation, Operation> ret;
  std::unordered_map<Tensor, Tensor> tmap;
  Map<Var, Var> vmap;
  for (const auto& op : ops) {
    Array<IterVar> ivs = op->root_iter_vars();
    DataType dtype = GetPromotedDataType(ivs);
    Operation new_op;
    const auto* compute = op.as<ComputeOpNode>();
    if (compute) {
      auto axis_func = [&vmap, dtype](const IterVar& iv) {
        IterVar new_iv = cast(dtype, iv);
        vmap.Set(iv->var, new_iv->var);
        return new_iv;
      };
      auto body_func = [&vmap, dtype](const PrimExpr& e) {
        DataTypeRewriter rewriter(vmap, dtype);
        return rewriter(e);
      };
      Array<IterVar> new_ivs
        = UpdateArray(compute->axis, axis_func);
      Array<PrimExpr> new_body
        = UpdateArray(compute->body, body_func);
      new_op = ComputeOp(compute->name, compute->tag, compute->attrs, new_ivs, new_body);
    } else {
      new_op = op;
    }
    new_op = new_op->ReplaceInputs(new_op, tmap);
    for (int i = 0; i < op->num_outputs(); ++i) {
      tmap[op.output(i)] = new_op.output(i);
    }
    ret.Set(op, new_op);
  }
  return ret;
}

}  // namespace te
}  // namespace tvm
