#include "HostMatrix.H"

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
