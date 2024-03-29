import re
import numpy as np
from scipy.sparse import coo_matrix

meaningless_name_list = [
    "frame_dummy",
    "call_weak_fn",
    "register_tm_clones",
    "deregister_tm_clones",
    "__do_global_dtors_aux",
    "__libc_csu_init",
    "__libc_csu_fini",
    "__x86.get_pc_thunk.bx",
    "__x86.get_pc_thunk.cx",
    "__x86.get_pc_thunk.dx",
    "__x86.get_pc_thunk.ax",
    "__x86.get_pc_thunk.si",
    "__x86.get_pc_thunk.di",
    "__x86.get_pc_thunk.bp",
    "entry",
    "_entry",
    "start",
    "_start",
    "init",
    "_init",
    "fini",
    "_fini",
]
def is_meaningful_funcname(name):
    if re.search(r"(sub|SUB)_[0-9a-fA-F]+", name):
        return False
    elif name in meaningless_name_list:
        return False
    else:
        return True

def filter_funcname_in_code(code, name, strip_name):
    if type(code) == str:
        new_code = code.replace(name,"<FUNCTION>")
        new_code = new_code.replace(strip_name,"<FUNCTION>")
    elif type(code) == list:
        new_code = []
        for inst in code:
            if name in inst:
                inst = inst.replace(name,"<FUNCTION>")
            if strip_name in inst:
                inst = inst.replace(strip_name,"<FUNCTION>")
            new_code.append(inst)
    else:
        raise Exception("code type error")
    return new_code

def str_to_scipy_sparse(mat_str):
    """
    Convert the string in input into a scipy sparse coo matrix. It uses a
    custom str encoding to save space and it's particularly suited for
    sparse matrices, like the graph adjacency matrix.

    Args:
        mat_str: string that encodes a numpy matrix

    Return
        numpy matrix
    """
    row_str, col_str, data_str, n_row, n_col = mat_str.split("::")

    n_row = int(n_row)
    n_col = int(n_col)

    # There are some cases where the BBs are not connected
    # Luckily they are not frequent (~10 among all 10**5 functions)
    if row_str == "" \
            or col_str == "" \
            or data_str == "":
        return np.identity(n_col)

    row = [int(x) for x in row_str.split(";")]
    col = [int(x) for x in col_str.split(";")]
    data = [int(x) for x in data_str.split(";")]
    np_mat = coo_matrix((data, (row, col)),
                        shape=(n_row, n_col)).toarray()
    return np_mat