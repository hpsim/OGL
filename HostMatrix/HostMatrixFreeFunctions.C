#include "HostMatrix.H"

namespace Foam {

const lduInterfaceField *interface_getter(
    const lduInterfaceFieldPtrsList &interfaces, const label i)
{
#ifdef WITH_ESI_VERSION
    return interfaces.get(i);
#else
    return interfaces.operator()(i);
#endif
}


// free functions - for unit testing
void symmetric_update(const label total_nnz, const label upper_nnz,
                      const label *permute, const scalar scale,
                      const scalar *diag, const scalar *upper, scalar *out)
{
    for (label i = 0; i < total_nnz; ++i) {
        const label pos{permute[i]};
        out[i] =
            scale * (pos >= upper_nnz) ? diag[pos - upper_nnz] : upper[pos];
    }
}

void symmetric_update_w_interface(const label total_nnz, const label diag_nnz,
                                  const label upper_nnz, const label *permute,
                                  const scalar scale, const scalar *diag,
                                  const scalar *upper, const scalar *interface,
                                  scalar *dense)
{
    for (label i = 0; i < total_nnz; ++i) {
        // where the element is stored in a combined array
        const label pos{permute[i]};
        scalar value;
        // all values up to upper_nnz_ are upper_nnz_ values
        if (pos < upper_nnz) {
            value = upper[pos];
        }
        // all values up to upper_nnz_ are upper_nnz_ values
        if (pos >= upper_nnz && pos < upper_nnz + diag_nnz) {
            value = diag[pos - upper_nnz];
        }
        if (pos >= upper_nnz + diag_nnz) {
            value = -interface[pos - upper_nnz - diag_nnz];
        }
        dense[i] = scale * value;
    }
}

void non_symmetric_update_w_interface(const label total_nnz,
                                      const label diag_nnz,
                                      const label upper_nnz,
                                      const label *permute, const scalar scale,
                                      const scalar *diag, const scalar *upper,
                                      const scalar *lower,
                                      const scalar *interface, scalar *dense)
{
    for (label i = 0; i < total_nnz; ++i) {
        const label pos{permute[i]};
        scalar value;
        if (pos < upper_nnz) {
            value = upper[pos];
        }
        if (pos >= upper_nnz && pos < 2 * upper_nnz) {
            value = lower[pos - upper_nnz];
        }
        if (pos >= 2 * upper_nnz && pos < 2 * upper_nnz + diag_nnz) {
            value = diag[pos - 2 * upper_nnz];
        }
        if (pos >= 2 * upper_nnz + diag_nnz) {
            value = -interface[pos - 2 * upper_nnz - diag_nnz];
        }
        dense[i] = scale * value;
    }
}


void non_symmetric_update(const label total_nnz, const label upper_nnz,
                          const label *permute, const scalar scale,
                          const scalar *diag, const scalar *upper,
                          const scalar *lower, scalar *dense)
{
    for (label i = 0; i < total_nnz; ++i) {
        const label pos{permute[i]};
        if (pos < upper_nnz) {
            dense[i] = scale * upper[pos];
            continue;
        }
        if (pos >= upper_nnz && pos < 2 * upper_nnz) {
            dense[i] = scale * lower[pos - upper_nnz];
            continue;
        }
        dense[i] = scale * diag[pos - 2 * upper_nnz];
    }
}

void init_local_sparsity(const label nrows, const label upper_nnz,
                         const bool is_symmetric, const label *upper,
                         const label *lower, label *rows, label *cols,
                         label *permute)
{
    label after_neighbours = (is_symmetric) ? upper_nnz : 2 * upper_nnz;
    label element_ctr = 0;
    label upper_ctr = 0;
    std::vector<std::vector<std::pair<label, label>>> lower_stack(upper_nnz);
    for (label row = 0; row < nrows; row++) {
        // add lower elements
        // for now just scan till current upper ctr
        for (const auto &[stored_upper_ctr, col] : lower_stack[row]) {
            rows[element_ctr] = row;
            cols[element_ctr] = col;
            permute[element_ctr] = stored_upper_ctr;
            element_ctr++;
        }

        // add diagonal elements
        rows[element_ctr] = row;
        cols[element_ctr] = row;
        permute[element_ctr] = after_neighbours + row;
        element_ctr++;

        // add upper elements
        // these are the transpose of the lower elements which are stored in
        // row major order.
        label row_upper = lower[upper_ctr];
        while (upper_ctr < upper_nnz && row_upper == row) {
            const label col_upper = upper[upper_ctr];

            rows[element_ctr] = row_upper;
            cols[element_ctr] = col_upper;

            // insert into lower_stack
            lower_stack[col_upper].emplace_back(
                (is_symmetric) ? upper_ctr : upper_ctr + upper_nnz, row_upper);
            permute[element_ctr] = upper_ctr;

            element_ctr++;
            upper_ctr++;
            row_upper = lower[upper_ctr];
        }
    }
}


}  // namespace Foam
