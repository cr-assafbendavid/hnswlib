#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <type_traits>

#include "hnswlib.h"

namespace py = pybind11;
using namespace pybind11::literals; // needed to bring in _a literal

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc.)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
inline void ParallelFor(size_t start, size_t end, size_t numThreads, const std::function<void(size_t, size_t)> &fn) {
    if (numThreads == 0)
        numThreads = std::thread::hardware_concurrency();
    if (numThreads == 1 || end - start < 2) {
        for (size_t id = start; id < end; ++id)
            fn(id, 0);
        return;
    }

    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
        threads.emplace_back([&, threadId] {
            while (true) {
                size_t id = current.fetch_add(1);
                if (id >= end)
                    break;
                try {
                    fn(id, threadId);
                } catch (...) {
                    std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                    lastException = std::current_exception();
                    /* This will work even when current is the largest value that
                     * size_t can fit, because fetch_add returns the previous value
                     * before the increment (what will result in overflow
                     * and produce 0 instead of current + 1).
                     */
                    current = end;
                    break;
                }
            }
        });
    }
    for (auto &thread : threads)
        thread.join();
    if (lastException)
        std::rethrow_exception(lastException);
}

template <typename dtype>
py::array_t<dtype> make_array(dtype *data, std::initializer_list<size_t> shape) {
    return py::array_t<dtype>{shape, data, py::capsule(data, [](void *p) { delete[] reinterpret_cast<dtype *>(p); })};
}

inline void assert_true(bool expr, const std::string &msg) {
    if (!expr)
        throw std::runtime_error("Unpickle Error: " + msg);
}

template<typename T> std::string typeName();
template<> std::string typeName<float>() { return "float"; }
template<> std::string typeName<uint32_t>() { return "uint32"; }

template<typename dist_t, typename data_t>
class Index {
public:
    Index(const std::string &space_name, size_t dim) :
            space_name{space_name}, dim{dim}, appr_alg{nullptr}, ep_added{true}, index_inited{false},
            num_threads_default{std::thread::hardware_concurrency()}, default_ef{10}, seed{0}, cur_l{0} {
        normalize = space_name == "cosine";
        l2space = init_space();
        if (!l2space)
            throw std::runtime_error("Invalid space name: " + space_name);
    }

    static const int ser_version = 1; // serialization version

    std::string space_name;
    size_t dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    bool normalize;
    size_t num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t> *appr_alg;
    hnswlib::SpaceInterface<dist_t> *l2space;

    ~Index() {
        delete l2space;
        delete appr_alg;
    }

    static inline std::string name() {
        return "HNSWIndex<" + typeName<dist_t>() + ", " + typeName<data_t>() + ">";
    }

    void init_new_index(const size_t maxElements, const size_t M, const size_t efConstruction,
                        const size_t random_seed) {
        if (appr_alg)
            throw std::runtime_error("The index is already initiated.");
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }

    void set_ef(size_t ef) {
        default_ef = ef;
        if (appr_alg)
            appr_alg->ef_ = ef;
    }

    void set_num_threads(int num_threads) {
        if (num_threads <= 0)
            throw std::runtime_error("number of threads must be positive");
        num_threads_default = num_threads;
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (appr_alg) {
            std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated.";
            delete appr_alg;
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements);
        cur_l = appr_alg->cur_element_count;
    }

    void normalize_vector(const data_t *data, float *norm_array) const {
        float norm = 0.0f;
        for (size_t i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (size_t i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }

    void addItems(const py::array_t<data_t, py::array::c_style | py::array::forcecast> &items,
                  const py::object &labels = py::none(), int num_threads = -1) {
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows, features;

        if (items.ndim() == 2) {
            rows = items.shape(0);
            features = items.shape(1);

        } else if (items.ndim() == 1) {
            rows = 1;
            features = items.shape(0);

        } else {
            throw std::runtime_error("data must be a 1d/2d array");
        }
        
        if (features != dim)
            throw std::runtime_error("wrong dimensionality of the vectors");

        // avoid using threads when the number of searches is small:
        if (rows <= (size_t)num_threads * 4)
            num_threads = 1;

        std::vector<size_t> ids;

        if (!labels.is_none()) {
            py::array_t<size_t> ids_arr(labels);
            if (ids_arr.ndim() == 1 && (size_t)ids_arr.shape(0) == rows) {
                ids.resize(rows);
                for (size_t i = 0; i < rows; ++i)
                    ids[i] = ids_arr.at(i);
            }
            else if (ids_arr.ndim() == 0 && rows == 1) {
                ids.push_back(*ids_arr.data());
            }
            else
                throw std::runtime_error("wrong dimensionality of the labels");
        }

        {
            py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
            int start = 0;
            if (!ep_added) {
                size_t id = ids.empty() ? cur_l : ids.at(0);
                auto vector_data = items.data(0);
                if (normalize) {
                    std::vector<float> norm_array(dim);
                    normalize_vector(vector_data, norm_array.data());
                    appr_alg->addPoint(norm_array.data(), id);
                } else {
                    appr_alg->addPoint(vector_data, id);
                }
                start = 1;
                ep_added = true;
            }

            py::gil_scoped_release l;
            if (!normalize) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.empty() ? cur_l + row : ids.at(row);
                    appr_alg->addPoint(items.data(row), id);
                });
            } else {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector(items.data(row), norm_array.data() + start_idx);
                    size_t id = ids.empty() ? cur_l + row : ids.at(row);
                    appr_alg->addPoint(norm_array.data() + start_idx, id);
                });
            }
            cur_l += rows;
        }
    }

    std::vector<std::vector<data_t>> getDataReturnList(const py::object &labelsobj = py::none()) {
        std::vector<std::vector<data_t>> data;
        if (!labelsobj.is_none()) {
            py::array_t<hnswlib::labeltype> labels(labelsobj);
            data.resize(labels.shape(0));

            for (size_t i = 0; i < data.size(); ++i)
                data[i] = appr_alg->template getDataByLabel<data_t>(labels.at(i));
        }
        return data;
    }

    std::vector<data_t> getData(hnswlib::labeltype id) {
        return appr_alg->template getDataByLabel<data_t>(id);
    }

    std::vector<hnswlib::labeltype> getIdsList() const {
        std::vector<hnswlib::labeltype> ids;
        for (const auto &kv : appr_alg->label_lookup_)
            if (!appr_alg->isMarkedDeleted(kv.second))
                ids.push_back(kv.first);
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    size_t size() const {
        size_t n = 0;
        for (auto kv : appr_alg->label_lookup_)
            if (!appr_alg->isMarkedDeleted(kv.second))
                ++n;
        return n;
    }

    py::dict getAnnData() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        unsigned int level0_npy_size = appr_alg->cur_element_count * appr_alg->size_data_per_element_;
        unsigned int link_npy_size = 0;
        std::vector<unsigned int> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++){
            unsigned int linkListSize = appr_alg->element_levels_[i] > 0 ?
                    appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i]=link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        auto* data_level0_npy = (char *) malloc(level0_npy_size);
        auto* link_list_npy = (char *) malloc(link_npy_size);
        auto* element_levels_npy = (int *) malloc(appr_alg->element_levels_.size()*sizeof(int));

        auto* label_lookup_key_npy = (hnswlib::labeltype *) malloc(appr_alg->label_lookup_.size()*sizeof(hnswlib::labeltype));
        auto* label_lookup_val_npy = (hnswlib::tableint *)  malloc(appr_alg->label_lookup_.size()*sizeof(hnswlib::tableint));

        memset(label_lookup_key_npy, -1, appr_alg->label_lookup_.size()*sizeof(hnswlib::labeltype));
        memset(label_lookup_val_npy, -1, appr_alg->label_lookup_.size()*sizeof(hnswlib::tableint));

        size_t idx=0;
        for ( auto it = appr_alg->label_lookup_.begin(); it != appr_alg->label_lookup_.end(); ++it ){
            label_lookup_key_npy[idx]= it->first;
            label_lookup_val_npy[idx]= it->second;
            idx++;
        }

        memset(link_list_npy, 0, link_npy_size);

        memcpy(data_level0_npy, appr_alg->data_level0_memory_, level0_npy_size);
        memcpy(element_levels_npy, appr_alg->element_levels_.data(), appr_alg->element_levels_.size() * sizeof(int));

        for (size_t i = 0; i < appr_alg->cur_element_count; i++){
            unsigned int linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize){
                memcpy(link_list_npy+link_npy_offsets[i], appr_alg->linkLists_[i], linkListSize);
            }
        }

        /*  TODO: serialize state of random generators appr_alg->level_generator_ and appr_alg->update_probability_generator_  */
        /*        for full reproducibility / to avoid re-initializing generators inside Index::createFromParams         */
        return py::dict(
                "offset_level0"_a=appr_alg->offsetLevel0_,
                "max_elements"_a=appr_alg->max_elements_,
                "cur_element_count"_a=appr_alg->cur_element_count,
                "size_data_per_element"_a=appr_alg->size_data_per_element_,
                "label_offset"_a=appr_alg->label_offset_,
                "offset_data"_a=appr_alg->offsetData_,
                "max_level"_a=appr_alg->maxlevel_,
                "enterpoint_node"_a=appr_alg->enterpoint_node_,
                "max_M"_a=appr_alg->maxM_,
                "max_M0"_a=appr_alg->maxM0_,
                "M"_a=appr_alg->M_,
                "mult"_a=appr_alg->mult_,
                "ef_construction"_a=appr_alg->ef_construction_,
                "ef"_a=appr_alg->ef_,
                "has_deletions"_a=appr_alg->has_deletions_,
                "size_links_per_element"_a=appr_alg->size_links_per_element_,
                "label_lookup_external"_a=make_array(label_lookup_key_npy, {appr_alg->label_lookup_.size()}),
                "label_lookup_internal"_a=make_array(label_lookup_val_npy, {appr_alg->label_lookup_.size()}),
                "element_levels"_a=make_array(element_levels_npy, {appr_alg->element_levels_.size()}),
                "data_level0"_a=make_array(data_level0_npy,{level0_npy_size}),
                "link_lists"_a=make_array(link_list_npy,{link_npy_size})
        );
    }

    py::dict getIndexParams() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        auto params = py::dict(
                "ser_version"_a=py::int_(ser_version), // serialization version
                "space"_a=space_name,
                "dim"_a=dim,
                "index_inited"_a=index_inited,
                "ep_added"_a=ep_added,
                "normalize"_a=normalize,
                "num_threads"_a=num_threads_default,
                "seed"_a=seed
        );
        if (!index_inited)
            return py::dict(**params, "ef"_a=default_ef);
        auto ann_params = getAnnData();
        return py::dict(**params, **ann_params);
    }

    static Index<dist_t, data_t> *createFromParams(const py::dict d) {
        // check serialization version
        assert_true(((int)py::int_(ser_version)) >= d["ser_version"].cast<int>(),
                "Invalid serialization version!");

        auto space_name_ = d["space"].cast<std::string>();
        auto dim_ = d["dim"].cast<int>();
        auto index_inited_ = d["index_inited"].cast<bool>();

        auto *new_index = new Index<dist_t, data_t>(space_name_, dim_);

        /*  TODO: deserialize state of random generators into new_index->level_generator_ and new_index->update_probability_generator_  */
        /*        for full reproducibility / state of generators is serialized inside Index::getIndexParams                      */
        new_index->seed = d["seed"].cast<size_t>();

        if (index_inited_) {
            new_index->appr_alg = new hnswlib::HierarchicalNSW<dist_t>(
                    new_index->l2space,d["max_elements"].cast<size_t>(), d["M"].cast<size_t>(),
                            d["ef_construction"].cast<size_t>(), new_index->seed);
            new_index->cur_l = d["cur_element_count"].cast<size_t>();
        }

        new_index->index_inited = index_inited_;
        new_index->ep_added = d["ep_added"].cast<bool>();
        new_index->num_threads_default = d["num_threads"].cast<int>();
        new_index->default_ef = d["ef"].cast<size_t>();

        if (index_inited_)
            new_index->setAnnData(d);

        return new_index;
    }

    static Index<dist_t, data_t> *createFromIndex(const Index<dist_t, data_t> &index) {
        return createFromParams(index.getIndexParams());
    }

    void setAnnData(const py::dict &d) { /* WARNING: Index::setAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        assert_true(appr_alg->offsetLevel0_ == d["offset_level0"].cast<size_t>(), "Invalid value of offsetLevel0_ ");
        assert_true(appr_alg->max_elements_ == d["max_elements"].cast<size_t>(), "Invalid value of max_elements_ ");

        appr_alg->cur_element_count = d["cur_element_count"].cast<size_t>();

        assert_true(appr_alg->size_data_per_element_ == d["size_data_per_element"].cast<size_t>(), "Invalid value of size_data_per_element_ ");
        assert_true(appr_alg->label_offset_ == d["label_offset"].cast<size_t>(), "Invalid value of label_offset_ ");
        assert_true(appr_alg->offsetData_ == d["offset_data"].cast<size_t>(), "Invalid value of offsetData_ ");

        appr_alg->maxlevel_ = d["max_level"].cast<int>();
        appr_alg->enterpoint_node_ = d["enterpoint_node"].cast<hnswlib::tableint>();

        assert_true(appr_alg->maxM_ == d["max_M"].cast<size_t>(), "Invalid value of maxM_ ");
        assert_true(appr_alg->maxM0_ == d["max_M0"].cast<size_t>(), "Invalid value of maxM0_ ");
        assert_true(appr_alg->M_ == d["M"].cast<size_t>(), "Invalid value of M_ ");
        assert_true(appr_alg->mult_ == d["mult"].cast<double>(), "Invalid value of mult_ ");
        assert_true(appr_alg->ef_construction_ == d["ef_construction"].cast<size_t>(), "Invalid value of ef_construction_ ");

        appr_alg->ef_ = d["ef"].cast<size_t>();
        appr_alg->has_deletions_=d["has_deletions"].cast<bool>();

        assert_true(appr_alg->size_links_per_element_ == d["size_links_per_element"].cast<size_t>(), "Invalid value of size_links_per_element_ ");

        auto label_lookup_key_npy = d["label_lookup_external"].cast<py::array_t<hnswlib::labeltype, py::array::c_style | py::array::forcecast>>();
        auto label_lookup_val_npy = d["label_lookup_internal"].cast<py::array_t<hnswlib::tableint, py::array::c_style | py::array::forcecast>>();
        auto element_levels_npy = d["element_levels"].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        auto data_level0_npy = d["data_level0"].cast<py::array_t<char, py::array::c_style | py::array::forcecast>>();
        auto link_list_npy = d["link_lists"].cast<py::array_t<char, py::array::c_style | py::array::forcecast>>();

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            if (label_lookup_val_npy.data()[i] < 0) {
                throw std::runtime_error("internal id cannot be negative!");
            } else {
                appr_alg->label_lookup_.insert(std::make_pair(label_lookup_key_npy.data()[i], label_lookup_val_npy.data()[i]));
            }
        }

        memcpy(appr_alg->element_levels_.data(), element_levels_npy.data(), element_levels_npy.nbytes());

        unsigned int link_npy_size = 0;
        std::vector<unsigned int> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            unsigned int linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        memcpy(appr_alg->data_level0_memory_, data_level0_npy.data(), data_level0_npy.nbytes());

        for (size_t i = 0; i < appr_alg->max_elements_; i++) {
            unsigned int linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize == 0) {
                appr_alg->linkLists_[i] = nullptr;
            } else {
                appr_alg->linkLists_[i] = (char *) malloc(linkListSize);
                if (appr_alg->linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");

                memcpy(appr_alg->linkLists_[i], link_list_npy.data()+link_npy_offsets[i], linkListSize);

            }
        }
    }

    py::tuple knnQuery_return_numpy(const py::object &input, size_t k = 1, int num_threads = -1) {
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows, k_eff;
        std::function<std::pair<size_t, const data_t *>(size_t)> get_data;

        if (input.is_none()) {
            auto labels = std::shared_ptr<std::vector<hnswlib::labeltype>>(
                    new std::vector<hnswlib::labeltype>(std::move(getIdsList())));
            rows = labels->size();
            get_data = [&, labels](size_t i) {
                auto label = labels->at(i);
                auto data = reinterpret_cast<const data_t *>(appr_alg->getDataByInternalId(
                        appr_alg->label2InternalId(label)));
                return std::make_pair(label, data);
            };
            k_eff = k + 1; // get k+1 neighbors and later remove "self"

        } else {
            auto items = std::make_shared<py::array_t<data_t, py::array::c_style | py::array::forcecast>>(input);
            size_t features;

            if (items->ndim() == 2) {
                rows = items->shape(0);
                features = items->shape(1);

            } else if (items->ndim() == 1) {
                rows = 1;
                features = items->shape(0);

            } else {
                throw std::runtime_error("data must be a 1d/2d array");
            }

            if (features != dim)
                throw std::runtime_error("wrong dimensionality for data");

            get_data = [items](size_t i) { return std::make_pair(i, items->data(i)); };
            k_eff = k;
        }

        hnswlib::labeltype *data_numpy_l;
        dist_t *data_numpy_d;
        {
            py::gil_scoped_release nogil;

            // avoid using threads when the number of searches is small
            if (rows <= (size_t)num_threads * 4)
                num_threads = 1;

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            std::vector<float> normed_data(normalize? num_threads * dim : 0);
            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                size_t label;
                const data_t *data;
                std::tie(label, data) = get_data(row);
                if (normalize) {
                    auto ndata = normed_data.data() + threadId * dim;
                    normalize_vector(data, ndata);
                    data = reinterpret_cast<const data_t *>(ndata);
                }
                auto result = appr_alg->searchKnn(data, k_eff);
                if (result.size() != k_eff)
                    throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                for (int i = (int)((row + 1) * k - 1); !result.empty(); result.pop()) {
                    auto &res = result.top();
                    if (k_eff == k || res.second != label) {
                        if (i == (int)(row * k) - 1) {
                            // This means that 'self' is not among the neighbors.
                            // We need to discard the furthest neighbor, so must move them all.
                            for (auto j = (row + 1) * k - 1; j > row * k; --j) {
                                data_numpy_d[j] = data_numpy_d[j - 1];
                                data_numpy_l[j] = data_numpy_l[j - 1];
                            }
                            ++i;
                        }
                        data_numpy_d[i] = res.first;
                        data_numpy_l[i] = res.second;
                        --i;
                    }
                }
            });
        }
        return py::make_tuple(
                make_array(data_numpy_l, {rows, k}),
                make_array(data_numpy_d, {rows, k})
        );
    }

    void markDeleted(hnswlib::labeltype label) {
        appr_alg->markDelete(label);
    }

    void resizeIndex(size_t new_size) {
        appr_alg->resizeIndex(new_size);
    }

    py::array_t<dist_t> pairwiseDistance(const py::array_t<hnswlib::labeltype> &labels, int num_threads) const {
        if (num_threads <= 0)
            num_threads = num_threads_default;
        size_t N = labels.size();
        size_t M = 2 * N - 1;
        size_t M2 = M * M;
        py::array_t<dist_t> dist(N * (N - 1) / 2);
        {
            py::gil_scoped_release nogil;

            ParallelFor(0, dist.size(), num_threads, [&](size_t k, size_t threadId) {
                auto i = (size_t)((M - sqrt(M2 - 8 * k)) / 2);
                size_t j = i * (i + 3) / 2 + k - i * N + 1;
                auto id1 = appr_alg->label2InternalId(labels.at(i));
                auto id2 = appr_alg->label2InternalId(labels.at(j));
                dist.mutable_at(k) = appr_alg->calcDistance(
                        appr_alg->getDataByInternalId(id1),
                        appr_alg->getDataByInternalId(id2));
            });
        }
        return dist;
    }

private:
    hnswlib::SpaceInterface<dist_t> *init_space() const;
};

template<>
hnswlib::SpaceInterface<float> *Index<float, float>::init_space() const {
    if (space_name == "l2")
        return new hnswlib::L2Space(dim);
    if (space_name == "ip" || space_name == "cosine")
        return new hnswlib::InnerProductSpace(dim);
    return nullptr;
}

template<>
hnswlib::SpaceInterface<float> *Index<float, uint32_t>::init_space() const {
    if (space_name == "hamming")
        return new hnswlib::HammingSpace(dim);
    if (space_name == "normleven")
        return new hnswlib::LevenshteinSpace<float, uint32_t>(dim);
    return nullptr;
}

template<>
hnswlib::SpaceInterface<uint32_t> *Index<uint32_t, uint32_t>::init_space() const {
    if (space_name == "leven")
        return new hnswlib::LevenshteinSpace<uint32_t, uint32_t>(dim);
    return nullptr;
}

template <typename dist_t, typename data_t>
void bindIndex(py::module &module) {
    using Idx = Index<dist_t, data_t>;

    py::class_<Idx>(module, Idx::name().c_str())
            .def(py::init(&Idx::createFromParams), py::arg("params"))
            /* WARNING: Index::createFromIndex is not thread-safe with Index::addItems */
            .def(py::init(&Idx::createFromIndex), py::arg("index"))
            .def(py::init<const std::string &, size_t>(), py::arg("space"), py::arg("dim"))
            .def("init_index", &Idx::init_new_index, py::arg("max_elements"), py::arg("M")=16,
                 py::arg("ef_construction")=200, py::arg("random_seed")=100)
            .def("knn_query", &Idx::knnQuery_return_numpy, py::arg("data")=py::none(),
                 py::arg("k")=1, py::arg("num_threads")=-1)
            .def("add_items", &Idx::addItems, py::arg("data"), py::arg("ids") = py::none(),
                 py::arg("num_threads")=-1)
            .def("get_items", &Idx::getDataReturnList, py::arg("ids") = py::none())
            .def("__getitem__", &Idx::getData, py::arg("id"))
            .def("get_ids_list", &Idx::getIdsList)
            .def("set_ef", &Idx::set_ef, py::arg("ef"))
            .def("set_num_threads", &Idx::set_num_threads, py::arg("num_threads"))
            .def("save_index", &Idx::saveIndex, py::arg("path_to_index"))
            .def("load_index", &Idx::loadIndex, py::arg("path_to_index"),
                 py::arg("max_elements")=0)
            .def("mark_deleted", &Idx::markDeleted, py::arg("label"))
            .def("resize_index", &Idx::resizeIndex, py::arg("new_size"))
            .def("pairwise_distance", &Idx::pairwiseDistance, py::arg("labels"), py::arg("num_threads")=-1)
            .def("__len__", &Idx::size)
            .def_readonly("space", &Idx::space_name)
            .def_readonly("dim", &Idx::dim)
            .def_readwrite("num_threads", &Idx::num_threads_default)
            .def_property("ef",
                  [](const Idx &index) {
                      return index.index_inited ? index.appr_alg->ef_ : index.default_ef;
                  },
                  [](Idx &index, const size_t ef_) {
                      index.default_ef=ef_;
                      if (index.appr_alg)
                          index.appr_alg->ef_ = ef_;
                  })
            .def_property_readonly("max_elements", [](const Idx &index) {
                return index.index_inited ? index.appr_alg->max_elements_ : 0;
            })
            .def_property_readonly("element_count", [](const Idx &index) {
                return index.index_inited ? index.appr_alg->cur_element_count : 0;
            })
            .def_property_readonly("ef_construction", [](const Idx &index) {
                return index.index_inited ? index.appr_alg->ef_construction_ : 0;
            })
            .def_property_readonly("M",  [](const Idx &index) {
                return index.index_inited ? index.appr_alg->M_ : 0;
            })
            .def(py::pickle(
                    [](const Idx &ind) { // __getstate__
                        // Return dict (wrapped in a tuple) that fully encodes state of the Index object
                        return py::make_tuple(ind.getIndexParams());
                    },
                    [](py::tuple &t) { // __setstate__
                        if (t.size() != 1)
                            throw std::runtime_error("Invalid state!");
                        return Idx::createFromParams(t[0].cast<py::dict>());
                    }
            ))
            .def("__repr__", [&](const Idx &ind) {
                return ind.name() + "(space='" + ind.space_name + "', dim=" + std::to_string(ind.dim) + ")";
            });
}


PYBIND11_MODULE(hnswlib, module) {
    module.def("Index", [](const std::string &space, const size_t dim) {
        if (space == "hamming" || space == "normleven")
            return py::cast(new Index<float, uint32_t>(space, dim),
                    py::return_value_policy::take_ownership);
        if (space == "leven")
            return py::cast(new Index<uint32_t, uint32_t>(space, dim),
                    py::return_value_policy::take_ownership);
        return py::cast(new Index<float, float>(space, dim),
                py::return_value_policy::take_ownership);
        }, py::arg("space"), py::arg("dim"));

    bindIndex<float, float>(module);
    bindIndex<float, uint32_t>(module);
    bindIndex<uint32_t, uint32_t>(module);
}
