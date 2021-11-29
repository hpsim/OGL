
#include <tuple>
#include "IOSortingIdxHandler.H"


namespace Foam {

IOSortingIdxHandler::IOSortingIdxHandler(const objectRegistry &db,
                                         const label nElems, const bool sort)
    : nElems_(nElems), is_sorted_(false), sort_(sort)
{
    // check object registry if sorting idxs exists
    word sorting_idxs_name = "sorting_idxs_";
    // TODO move this to CTR
    if (db.foundObject<IOField<label>>(sorting_idxs_name)) {
        IOField<label> &labelListRef =
            db.lookupObjectRef<IOField<label>>(sorting_idxs_name);
        sorting_idxs_ = &labelListRef;
        set_is_sorted(true);
    } else {
        const fileName path = sorting_idxs_name;
        sorting_idxs_ = new IOField<label>(IOobject(path, db));
        init_sorting_idxs();
        set_is_sorted(false);
    }
};


void IOSortingIdxHandler::compute_sorting_idxs(
    const std::shared_ptr<idx_array> row_idxs,
    const std::shared_ptr<idx_array> col_idxs, const label nCells)
{
    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    std::stable_sort(
        sorting_idxs_->data(), &sorting_idxs_->data()[nElems_],
        [this, &row_idxs, &col_idxs, nCells](size_t i1, size_t i2) {
            return std::tie(row_idxs->get_data()[i1],
                            col_idxs->get_data()[i1]) <
                   std::tie(row_idxs->get_data()[i2], col_idxs->get_data()[i2]);
        });

    std::vector<label> tmp_sorting_idxs(nElems_);
    for (label i = 0; i < nElems_; i++) {
        tmp_sorting_idxs[i] = sorting_idxs_->operator[](i);
    }

    for (label i = 0; i < nElems_; i++) {
        sorting_idxs_->operator[](tmp_sorting_idxs[i]) = i;
    }
};

void IOSortingIdxHandler::init_sorting_idxs()
{
    // initialize original index locations
    sorting_idxs_->resize(nElems_);

    // set sorting idxs from 0 .. n
    for (int i = 0; i < nElems_; i++) {
        sorting_idxs_->operator[](i) = i;
    }
};


}  // namespace Foam
