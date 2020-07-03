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
 * \file iter_var_type_promotion.h
 */
#ifndef TVM_TE_SCHEDULE_ITER_VAR_TYPE_PROMOTION_H_
#define TVM_TE_SCHEDULE_ITER_VAR_TYPE_PROMOTION_H_
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

DataType GetTargetDataType(Array<IterVar> vars);

IterVar MakeIterVar(DataType dtype, IterVar iv);

IterVar UpdateIterVar(Map<Var, IterVar>* vmap, DataType dtype, IterVar iv) {
    if (vmap->find(iv->var) != vmap->end()) {
      return vmap->at(iv->var);
    }
    IterVar new_iv = MakeIterVar(dtype, iv);
    vmap->Set(iv->var, new_iv);
    return new_iv;
}

class DataTypeRewriter : public ExprMutator {
 public:
  DataTypeRewriter(Map<Var, IterVar>* vmap, DataType target_dtype): 
    vmap_(vmap), target_dtype_(target_dtype) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    if (vmap_->find(var) == vmap_->end()) {
      return var;
    }
    return vmap_->at(var)->var;
  }

  PrimExpr VisitExpr_(const ReduceNode* op) final {
    std::vector<IterVar> ivs;
    ivs.reserve(op->axis.size());
    for (const auto& iv : op->axis) {
      IterVar new_iv;
      if (vmap_->find(iv->var) == vmap_->end()) {
        new_iv = MakeIterVar(target_dtype_, iv);
        vmap_->Set(iv->var, new_iv);
      } else {
        new_iv = vmap_->at(iv->var);
      }
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
  Map<Var, IterVar>* vmap_;
  DataType target_dtype_;
  // cached ops
  const Op& builtin_pow_ = Op::Get("tir.pow");
};

class IterVarRelationRewriter {
 public:
  IterVarRelationRewriter(Map<Var, IterVar>* vmap, DataType target_dtype): 
    vmap_(vmap), target_dtype_(target_dtype) {}

  IterVarRelation operator()(const IterVarRelation& ivrel) {
    if (const auto* rel = ivrel.as<SplitNode>()) {
      VisitRel_(rel);
    } else if (const auto* rel = ivrel.as<FuseNode>()) {
      VisitRel_(rel);
    } else if (const auto* rel = ivrel.as<RebaseNode>()) {
      VisitRel_(rel);
    } else if (const auto* rel = ivrel.as<SingletonNode>()) {
      VisitRel_(rel);
    } 
    LOG(FATAL) << "Invalid IterVarRelation: " << ivrel->GetTypeKey();
    return IterVarRelation();
  }

  IterVarRelation VisitRel_(const SplitNode* rel) {
    return Split(
      UpdateIterVar(vmap_, target_dtype_, rel->parent),
      UpdateIterVar(vmap_, target_dtype_, rel->outer),
      UpdateIterVar(vmap_, target_dtype_, rel->inner),
      (rel->factor.defined() ? cast(target_dtype_, rel->factor) : rel->factor),
      (rel->nparts.defined() ? cast(target_dtype_, rel->nparts) : rel->nparts)
    );
  }

  IterVarRelation VisitRel_(const FuseNode* rel) {
    return Fuse(
      UpdateIterVar(vmap_, target_dtype_, rel->outer),
      UpdateIterVar(vmap_, target_dtype_, rel->inner),
      UpdateIterVar(vmap_, target_dtype_, rel->fused)
    );
  }

  IterVarRelation VisitRel_(const RebaseNode* rel) {
    return Rebase(
      UpdateIterVar(vmap_, target_dtype_, rel->parent),
      UpdateIterVar(vmap_, target_dtype_, rel->rebased)
    );
  }

  IterVarRelation VisitRel_(const SingletonNode* rel) {
    return Singleton(
      UpdateIterVar(vmap_, target_dtype_, rel->iter)
    );
  }
 private:
  Map<Var, IterVar>* vmap_;
  DataType target_dtype_;
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_SCHEDULE_ITER_VAR_TYPE_PROMOTION_H_
