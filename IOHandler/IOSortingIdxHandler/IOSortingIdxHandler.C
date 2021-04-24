
#include "IOSortingIdxHandler.H"

namespace Foam {

IOSortingIdxHandler::IOSortingIdxHandler(const objectRegistry &db,
                                         const label nElems, const bool sort)
    : nElems_(nElems), is_sorted_(false), sort_(sort)
{
    // check object registry if sorting idxs exists
    word sorting_idxs_name = "sorting_idxs_";
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
    const std::vector<label> &row_idxs)
{
    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    std::stable_sort(sorting_idxs_->data(), &sorting_idxs_->data()[nElems_],
                     [this, row_idxs](size_t i1, size_t i2) {
                         return row_idxs[i1] < row_idxs[i2];
                     });
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
