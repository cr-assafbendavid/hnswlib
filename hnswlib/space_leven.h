#pragma once

#include "hnswlib.h"
#include "leven/leven/levenshtein_impl.h"

namespace hnswlib {
    template<typename data_t>
    void levendist(const data_t *str1, uint32_t len1, const data_t *str2, uint32_t len2, uint32_t &dist) {
        dist = levenshtein(str1, len1, str2, len2);
    }

    template<typename data_t>
    void levendist(const data_t *str1, uint32_t len1, const data_t *str2, uint32_t len2, float &dist) {
        dist = static_cast<float>(levenshtein(str1, len1, str2, len2));
        auto size = std::max(len1, len2);
        if (size)
            dist /= static_cast<float>(size);
    }

    template<typename dist_t, typename data_t>
    static dist_t levenshtein_distance(const void *p1, const void *p2, const void *pdim) {
        const auto *str1 = reinterpret_cast<const data_t *>(p1);
        const auto *str2 = reinterpret_cast<const data_t *>(p2);
        const auto max_length = *reinterpret_cast<const uint32_t *>(pdim);

        auto len = [&](const data_t * const str) {
            auto p = str;
            auto const end = str + max_length;
            while (*p != 0 && p != end)
                ++p;
            return p - str;
        };

        dist_t dist;
        levendist(str1, len(str1), str2, len(str2), dist);
        return dist;
    }

    template<typename dist_t, typename data_t>
    class LevenshteinSpace : public SpaceInterface<dist_t> {
        size_t data_size_;
        uint32_t max_length_;

    public:
        explicit LevenshteinSpace(size_t max_length)
        : max_length_((uint32_t)max_length) {
            data_size_ = max_length * sizeof(data_t);
        }

        size_t get_data_size() override {
            return data_size_;
        }

        void *get_dist_func_param() override {
            return &max_length_;
        }

        DISTFUNC<dist_t> get_dist_func() override {
            return levenshtein_distance<dist_t, data_t>;
        }
    };
}
