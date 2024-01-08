#pragma once
#include "hnswlib.h"

namespace hnswlib {
    static float HammingDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const auto *pVect1 = reinterpret_cast<const uint32_t *>(pVect1v);
        const auto *pVect2 = reinterpret_cast<const uint32_t *>(pVect2v);
        const auto qty = *reinterpret_cast<const size_t *>(qty_ptr);

        int n_different = 0;
        for (size_t i = 0; i < qty; ++i)
            n_different += (int)(*pVect1++ != *pVect2++);
        return ((float)n_different) / qty;
    }

    class HammingSpace : public SpaceInterface<float> {
    private:
        DISTFUNC<float> distfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        explicit HammingSpace(size_t dim)
        : dim_(dim), distfunc_(HammingDist) {
            data_size_ = dim * sizeof(uint32_t);
        }

        size_t get_data_size() override {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() override {
            return distfunc_;
        }

        void *get_dist_func_param() override {
            return &dim_;
        }
    };
}
