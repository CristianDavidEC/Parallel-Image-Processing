CLASS_1 = [
    [0, 0,  0, 0, 0],
    [0, 0,  1, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0,  0, 0, 0],
    [0, 0,  0, 0, 0]
]

CLASS_2 = [
    [0, 0,  0, 0, 0],
    [0, 0,  0, 1, 0],
    [0, 0, -1, 0, 0],
    [0, 0,  0, 0, 0],
    [0, 0,  0, 0, 0],
]

CLASS_3 = [
    [0, 0, -1, 0, 0],
    [0, 0,  3, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0,  1, 0, 0],
    [0, 0,  0, 0, 0],
]

SQUARE_3X3 = [
    [0,  0,  0,  0, 0],
    [0, -1,  2, -1, 0],
    [0,  2, -4,  2, 0],
    [0, -1,  2, -1, 0],
    [0,  0,  0,  0, 0],
]

EDGE_3X3 = [
    [0,  0,  0,  0, 0],
    [0, -1,  2, -1, 0],
    [0,  2, -4,  2, 0],
    [0,  0,  0,  0, 0],
    [0,  0,  0,  0, 0],
]

SQUARE_5X5 = [
    [-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1],
]

EDGE_5X5 = [
    [-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 0,  0,   0,  0,  0],
    [ 0,  0,   0,  0,  0],
]

SOBLE_VERTICAL = [
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1],
]

SOBLE_HORIZONTAL = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
]

KERNEL_LAPLACE = [
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
]

PREWITT_VERTICAL = [
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1],
]

PREWITT_HORIZONTAL = [
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
]

KERNELS_DICT = {
    "CLASS_1": CLASS_1,
    "CLASS_2": CLASS_2,
    "CLASS_3": CLASS_3,
    "SQUARE_3X3": SQUARE_3X3,
    "EDGE_3X3": EDGE_3X3,
    "SQUARE_5X5": SQUARE_5X5,
    "EDGE_5X5": EDGE_5X5,
    "SOBLE_VERTICAL": SOBLE_VERTICAL,
    "SOBLE_HORIZONTAL": SOBLE_HORIZONTAL,
    "KERNEL_LAPLACE": KERNEL_LAPLACE,
    "PREWITT_VERTICAL": PREWITT_VERTICAL,
    "PREWITT_HORIZONTAL": PREWITT_HORIZONTAL,
}