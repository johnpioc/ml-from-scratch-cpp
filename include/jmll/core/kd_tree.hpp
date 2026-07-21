#include <jmll/core/matrix.hpp>
#include <jmll/core/vector.hpp>
#include <memory>
#include <vector>

namespace jmll::core {
class KDNode {
   private:
    Vector key_;
    int dimIndex_ = 0;
    double value_;
    std::shared_ptr<KDNode> left_;
    std::shared_ptr<KDNode> right_;

   public:
    KDNode(const Vector& key, double value);

    void setLeft(std::shared_ptr<KDNode> node);
    std::shared_ptr<KDNode> getLeft();

    void setRight(std::shared_ptr<KDNode> node);
    std::shared_ptr<KDNode> getRight();

    [[nodiscard]] int getDimIndex() const;
    void setDimIndex(int dimIndex);

    [[nodiscard]] int getKeySize() const;
    [[nodiscard]] double getKeyVal(int i) const;
};

class KDTree {
   private:
    std::shared_ptr<KDNode> root_;
    int size_;

   public:
    KDTree(const Matrix& data, const Vector& labels);
};
}  // namespace jmll::core
