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
 * \file ad_utils.h
 * \brief Helper utilities to implement auto-differentiation.
 */
#ifndef TVM_TE_AUTODIFF_AD_UTILS_H_
#define TVM_TE_AUTODIFF_AD_UTILS_H_

#include <tvm/arith/int_solver.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace te {

class LinkedHashSetNode : public Object {
 public:
  /*! \brief Type of the keys in the hash map */
  using key_type = ObjectRef;
  using ComparatorType = bool(const key_type&, const key_type&);
  /*! \brief Type of the actual underlying container */
  using ContainerType = std::set<ObjectRef, ComparatorType>;
  /*! \brief Iterator class */
  using iterator = ContainerType::iterator;
  /*! \brief Iterator class */
  using const_iterator = ContainerType::const_iterator;

  static constexpr const char* _type_key = "LinkedHashSet";
  TVM_DECLARE_FINAL_OBJECT_INFO(LinkedHashSetNode, Object);

  /*!
   * \brief Number of elements in the LinkedHashSetNode
   * \return The result
   */
  size_t size() const { return data_.size(); }
  /*! \return begin iterator */
  iterator begin() { return data_.begin(); }
  /*! \return const begin iterator */
  const_iterator begin() const { return data_.begin(); }
  /*! \return end iterator */
  iterator end() { return data_.end(); }
  /*! \return end iterator */
  const_iterator end() const { return data_.end(); }
  /*!
   * \brief Iterator associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  const_iterator find(const key_type& key) const { return data_.find(key); }
  /*!
   * \brief Iterator associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  iterator find(const key_type& key) { return data_.find(key); }
  /*!
   * \brief Erase the entry associated with the iterator
   * \param position The iterator
   */
  void erase(const iterator& position) { data_.erase(position); }
  /*!
   * \brief Erase the entry associated with the key, do nothing if not exists
   * \param key The indexing key
   */
  void erase(const key_type& key) { data_.erase(key); }
  void insert(const key_type& key) {
    data_.insert(key);
    idx_.Set(key, IntImm(DataType::Int(32), cnt_++));
  }
  /*!
   * \brief Create an empty container
   * \return The object created
   */
  static ObjectPtr<MapNode> Empty() { return make_object<LinkedHashSetNode>(); }

 protected:
  /*!
   * \brief Create the set using contents from the given iterators.
   * \param first Begin of iterator
   * \param last End of iterator
   * \tparam IterType The type of iterator
   * \return ObjectPtr to the map created
   */
  template <typename IterType>
  static ObjectPtr<Object> CreateFromRange(IterType first, IterType last) {
    ObjectPtr<LinkedHashSetNode> p = make_object<LinkedHashSetNode>();
    p->data_ = ContainerType(first, last);
    for (auto it = first; it != last; ++it) {
      p->idx_[*it] = IntImm(DataType::Int(32), p->cnt_++);
    }
    return p;
  }

  // struct Comparator {
  //   bool operator() (const key_type& a, const key_type& b) {
  //     int64_t va = idx_.at(a)->value;
  //     int64_t vb = idx_.at(b)->value;
  //   }
  // };
 private:
  bool Comparator(const key_type& a, const key_type& b) {
    int64_t va = idx_.at(a)->value;
    int64_t vb = idx_.at(b)->value;
    return va < vb;
  }

  /*! \brief The real container storing data */
  ContainerType data_(Comparator);
  Map<ObjectRef, IntImm> idx_;
  int cnt_{0};
  template <typename, typename>
  friend class LinkedHashSet;
};

/*!
 * \brief Map container of NodeRef->NodeRef in DSL graph.
 *  Map implements copy on write semantics, which means map is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const acces, use Set to mutate the content.
 * \tparam K The key NodeRef type.
 * \tparam V The value NodeRef type.
 */
template <typename T,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type>
class LinkedHashSet : public ObjectRef {
 public:
  using key_type = K;
  using mapped_type = V;
  class iterator;
  /*!
   * \brief default constructor
   */
  Map() { data_ = MapNode::Empty(); }
  /*!
   * \brief move constructor
   * \param other source
   */
  Map(Map<K, V>&& other) { data_ = std::move(other.data_); }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Map(const Map<K, V>& other) : ObjectRef(other.data_) {}
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(Map<K, V>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(const Map<K, V>& other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Map(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Map(IterType begin, IterType end) {
    data_ = MapNode::CreateFromRange(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V>> init) {
    data_ = MapNode::CreateFromRange(init.begin(), init.end());
  }
  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template <typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    data_ = MapNode::CreateFromRange(init.begin(), init.end());
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V at(const K& key) const { return DowncastNoCheck<V>(GetMapNode()->at(key)); }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V operator[](const K& key) const { return this->at(key); }
  /*! \return The size of the array */
  size_t size() const {
    MapNode* n = GetMapNode();
    return n == nullptr ? 0 : n->size();
  }
  /*! \return The number of elements of the key */
  size_t count(const K& key) const {
    MapNode* n = GetMapNode();
    return n == nullptr ? 0 : GetMapNode()->count(key);
  }
  /*! \return whether array is empty */
  bool empty() const { return size() == 0; }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be setted.
   */
  void Set(const K& key, const V& value) {
    CopyOnWrite();
    MapNode::InsertMaybeReHash(MapNode::KVType(key, value), &data_);
  }
  /*! \return begin iterator */
  iterator begin() const { return iterator(GetMapNode()->begin()); }
  /*! \return end iterator */
  iterator end() const { return iterator(GetMapNode()->end()); }
  /*! \return find the key and returns the associated iterator */
  iterator find(const K& key) const { return iterator(GetMapNode()->find(key)); }

  void erase(const K& key) { CopyOnWrite()->erase(key); }

  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  MapNode* CopyOnWrite() {
    if (data_.get() == nullptr) {
      data_ = MapNode::Empty();
    } else if (!data_.unique()) {
      data_ = MapNode::CopyFrom(GetMapNode());
    }
    return GetMapNode();
  }
  /*! \brief specify container node */
  using ContainerType = MapNode;

  /*! \brief Iterator of the hash map */
  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = const std::pair<K, V>;
    using pointer = value_type*;
    using reference = value_type;

    iterator() : itr() {}

    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const { return itr == other.itr; }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return itr != other.itr; }
    /*! \brief De-reference iterators is not allowed */
    pointer operator->() const = delete;
    /*! \brief De-reference iterators */
    reference operator*() const {
      auto& kv = *itr;
      return std::make_pair(DowncastNoCheck<K>(kv.first), DowncastNoCheck<V>(kv.second));
    }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++() {
      ++itr;
      return *this;
    }
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      iterator copy = *this;
      ++(*this);
      return copy;
    }

   private:
    iterator(const MapNode::iterator& itr)  // NOLINT(*)
        : itr(itr) {}

    template <typename, typename, typename, typename>
    friend class Map;

    MapNode::iterator itr;
  };

 private:
  /*! \brief Return data_ as type of pointer of MapNode */
  MapNode* GetMapNode() const { return static_cast<MapNode*>(data_.get()); }
};


/*!
 * \brief Clone iter vars and return both the new vars and the substitution from old to new.
 *
 * \param vars The original iter vars.
 * \return A pair containing the array of new iter vars and the map from old vars to new ones.
 */
std::pair<Array<IterVar>, Map<Var, PrimExpr>> CloneIterVars(const Array<IterVar>& vars);

/*!
 * \brief Clone reduction by cloning the axis variables.
 * \param expr A reduction expr to clone. Non-reduction expressions are left intact.
 */
PrimExpr CloneReduction(const PrimExpr& expr);

/*!
 * \brief Create a tensor from an expression. The expression may be a reduction, in which
 *  case its body will be correctly duplicated if it is a multi-valued reduction.
 *
 * \param expr The expr which will be the tensor's body.
 * \param axis The input variables with ranges.
 * \param name The tensor's name.
 * \param tag The tensor's tag.
 * \param attrs The tensor's attrs.
 * \param clone_axis Whether to clone the given axis and perform substitution.
 * \return A tensor.
 */
Tensor TensorFromExpr(const PrimExpr& expr, const Array<IterVar>& axis,
                      const std::string& name = "tensor", const std::string& tag = "",
                      const Map<String, ObjectRef>& attrs = {}, bool clone_axis = true);

Tensor TransformTensorBody(
    const Tensor& tensor,
    const std::function<PrimExpr(const PrimExpr&, const Array<IterVar>&)>& func);

Tensor TransformTensorBody(const Tensor& tensor,
                           const std::function<PrimExpr(const PrimExpr&)>& func);

/*!
 * \brief Inline tensors access recursively.
 *
 *  This function will inline tensors recursively until it reaches a tensor which is impossible to
 *  inline (a reduction if \p inline_reductions is false, a non-compute tensor, a tensor which is
 *  not from \p inlineable). It won't descend into non-inlinable tensors' bodies.
 *
 * \param tensor The tensor whose body to transform.
 * \param inlineable A list of tensors which are allowed to be inlined. If empty, try
 *  to inline all tensors.
 * \param inline_reductions Whether to inline reductions (this may result in top-level reduction
 *  nodes).
 *
 * \return An inlined tensor
 */
TVM_DLL Tensor InlineTensorAccess(const Tensor& tensor,
                                  const Array<Tensor>& inlineable = Array<Tensor>(),
                                  bool inline_reductions = false);

/*!
 * \brief Inline tensors access at the tail.
 * \param tensor The tensor whose body to transform.
 * \return An inlined tensor
 */
TVM_DLL Tensor InlineTailTensorAccess(const Tensor& tensor);

/*!
 * \brief Simplify an iteration domain.
 *
 *  An iteration domain is basically an array of variables and a condition. The function will do the
 *  following:
 *  - Replace div and mod operations with new variables (optional).
 *  - Extract (in)equalities from the condition.
 *  - Perform Fourier-Motzkin elimination.
 *  - Shear the domain of iteration (e.g. if `y <= x <= y + 2` then x will be replaced with `y + d`
 *    where `d` is a new variable such that `0 <= d <= 2`).
 *  - Remove redundant variables.
 *  - Infer new variable ranges (hopefully more precise).
 *
 * \param iter_domains The original domain.
 * \param eliminate_div_mod Whether to eliminate div and mod by introducing new variables.
 */
TVM_DLL arith::IntConstraintsTransform SimplifyDomain(const arith::IntConstraints& iter_domains,
                                                      bool eliminate_div_mod = true);

/*!
 * \brief Perform lifting of conditions of being possible to be non-zero together with
 *  applying some transformations like simplifying the reduction domain. Works only with
 *  this particular tensor's body, i.e. doesn't perform inlining.
 *
 * \param tensor The original tensor;
 * \param vranges Optional map from free variables to their value ranges.
 * \return An optimized tensor.
 */
TVM_DLL Tensor RemoveJacobianAndLiftNonzeroCond(const Tensor& tensor,
                                                const Map<Var, Range>& vranges = Map<Var, Range>());

}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_AUTODIFF_AD_UTILS_H_
