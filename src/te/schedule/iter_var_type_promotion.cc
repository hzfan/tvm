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
#include "iter_var_type_promotion.h"

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

DataType GetTargetDataType(Array<IterVar> vars) {
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

IterVar MakeIterVar(DataType dtype, IterVar iv) {
  Var v = Var(iv->var->name_hint, dtype);
  Range dom = Range(cast(dtype, iv->dom->min), cast(dtype, iv->dom->extent));
  return IterVar(dom, v, iv->iter_type, iv->thread_tag);
}

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

  if (op->op.same_as(builtin::if_then_else())) {
    return if_then_else(op->args[0], op->args[1], op->args[2]);
  } else if (op->op.same_as(builtin::shift_right())) {
    return op->args[0] >> op->args[1];
  } else if (op->op.same_as(builtin::shift_left())) {
    return op->args[0] << op->args[1];
  } else if (op->op.same_as(builtin::bitwise_and())) {
    return op->args[0] & op->args[1];
  } else if (op->op.same_as(builtin::bitwise_or())) {
    return op->args[0] | op->args[1];
  } else if (op->op.same_as(builtin::bitwise_xor())) {
    return op->args[0] ^ op->args[1];
  } else if (op->op.same_as(builtin_pow_)) {
    return pow(op->args[0], op->args[1]);
  }

  return e;
}

IterVarRelation RewriteIterVarRelation(IterVarRelation ivrel, DataType dtype) {
  
IterVarRelation RewriteIterVarRelation(IterVarRelation ivrel, DataType dtype) {
  if (const auto* rel = ivrel.as<SplitNode>()) {

  } else if (const auto* rel = ivrel.as<FuseNode>()) {

  } else if (const auto* rel = ivrel.as<RebaseNode>()) {

  } else if (const auto* rel = ivrel.as<SingletonNode>()) {

  } 
  LOG(FATAL) << "Invalid IterVarRelation: " << ivrel->GetTypeKey();
  return IterVarRelation();
}

}  // namespace te
}  // namespace tvm
