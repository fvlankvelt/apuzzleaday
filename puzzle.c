#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BOARD_WIDTH 7
#define BOARD_HEIGHT 7

#define N_PIECES 8
#define N_DEFINITIONS 13

typedef unsigned char bool;
#define true ((bool) 1)
#define false ((bool) 0)

typedef char cell;

typedef struct _orientation {
  int height;
  int width;
  bool flipped;
  cell * value;
} orientation;

typedef struct _definition {
  bool mirror;
  int rotations;
  orientation orientation;
} definition;

typedef struct _piece {
  int n_orientations;
  orientation orientations[8];
} piece;

typedef struct _board {
  cell cells[BOARD_WIDTH * BOARD_HEIGHT];
  bool flips[N_PIECES];
} board;

typedef struct _rotation {
  int x_x;
  int x_y;
  int y_x;
  int y_y;
} rotation;

typedef struct _solution {
  board board;
  struct _solution * next;
} solution;

board initial_board;

static inline void board_set(board * board, int i, int j, cell value) {
  board->cells[j * BOARD_WIDTH + i] = value;
}

static inline cell board_get(board * board, int i, int j) {
  return board->cells[j * BOARD_WIDTH + i];
}

definition all_definitions[N_DEFINITIONS] = {
  { false, 2,
    { 2, 3, false, (cell[]) {
      'O', 'O', 'O',
      'O', 'O', 'O' } } },
  // 0
  { true, 4,
    { 2, 3, false, (cell[]) {
      'P', 'P', 0 ,
      'P', 'P', 'P' } } },
  // 1
  { false, 4,
    { 2, 3, false, (cell[]) {
      'U', 'U', 'U',
      'U',  0 , 'U' } } },
  // 2
  { false, 4,
    { 3, 3, false, (cell[]) {
      'V', 'V', 'V',
      'V',  0 ,  0 ,
      'V',  0 ,  0 , } } },
  // 3
  { true, 2,
    { 3, 3, false, (cell[]) {
      'Z', 'Z',  0 ,
       0 , 'Z',  0 ,
       0 , 'Z', 'Z', } } },
  // 4
  { true, 4,
    { 2, 4, false, (cell[]) {
      'L', 'L', 'L', 'L',
      'L',  0 ,  0 ,  0  } } },
  // 5
  { true, 4,
    { 2, 4, false, (cell[]) {
      'Y', 'Y', 'Y', 'Y',
       0 ,  0 , 'Y',  0  } } },
  // 6
  { true, 4,
    { 2, 4, false, (cell[]) {
       0 , 'N', 'N', 'N',
      'N', 'N',  0 ,  0  } } },
  // 7
  { false, 1,
    { 3, 3, false, (cell[]) {
       0 , 'X',  0 ,
      'X', 'X', 'X',
       0 , 'X',  0 } } },
  // 8
  { false, 4,
    { 3, 3, false, (cell[]) {
      'W', 'W',  0 ,
       0 , 'W', 'W',
       0 ,  0 , 'W'} } },
  // 9
  { false, 2,
    { 5, 1, false, (cell[]) {
      'I', 'I', 'I', 'I', 'I'} } },
  // 10
  { true, 4,
    { 3, 3, false, (cell[]) {
      'F', 'F',  0 ,
       0 , 'F', 'F',
       0 , 'F',  0 , } } },
  // 11
  { false, 4,
    { 3, 3, false, (cell[]) {
      'T', 'T', 'T',
       0 , 'T',  0 ,
       0 , 'T',  0 , } } },
};
piece all_pieces[N_DEFINITIONS];

piece pieces[N_PIECES];

rotation rotations[4] = {
  {  1,  0,  0,  1 },
  {  0,  1, -1,  0 },
  { -1,  0,  0, -1 },
  {  0, -1,  1,  0 },
};

#define N_COMBINATIONS 37


// 0:P 1:U 2:V 3:Z 4:L 5:Y 6:N 7:X 8:W 9:I 10:F 11:T
int combinations[][N_PIECES - 1] = {
  { 0, 1, 2, 3, 4, 5, 10 },   // 18872  7 205 P-U-V-Z-L-Y-F
  { 0, 1, 2, 3, 4, 5, 6 },    // 24405  7 216 P-U-V-Z-L-Y-N
  { 0, 1, 2, 3, 4, 5, 8 },    //  7281  1  97 P-U-V-Z-L-Y-W
  { 0, 1, 2, 3, 4, 5, 9 },    // 10300  2 114 P-U-V-Z-L-Y-I
  { 0, 1, 2, 3, 4, 6, 10 },   // 15667  3 169 P-U-V-Z-L-N-F
  { 0, 1, 2, 3, 4, 6, 8 },    //  6123  1  81 P-U-V-Z-L-N-W
  { 0, 1, 2, 3, 4, 9, 10 },   //  7088  2  86 P-U-V-Z-L-I-F
  { 0, 1, 2, 3, 5, 10, 11 },  //  5991  1  49 P-U-V-Z-Y-F-T // hard!
  { 0, 1, 2, 3, 5, 6, 10 },   // 11980  1 126 P-U-V-Z-Y-N-F
  { 0, 1, 2, 4, 5, 10, 11 },  // 19503  6 161 P-U-V-L-Y-F-T
  { 0, 1, 2, 4, 5, 6, 10 },   // 36485 19 434 P-U-V-L-Y-N-F // very many solutions
  { 0, 1, 2, 4, 5, 6, 11 },   // 22625 10 208 P-U-V-L-Y-N-T
  { 0, 1, 2, 4, 5, 6, 8 },    // 14664  1 189 P-U-V-L-Y-N-W
  { 0, 1, 2, 4, 5, 6, 9 },    // 20330  2 350 P-U-V-L-Y-N-I
  { 0, 1, 2, 4, 5, 8, 10 },   // 15105  3 275 P-U-V-L-Y-W-F
  { 0, 1, 2, 4, 5, 8, 11 },   //  7252  1 130 P-U-V-L-Y-W-T
  { 0, 1, 2, 4, 5, 9, 10 },   // 16154  6 244 P-U-V-L-Y-I-F
  { 0, 1, 2, 4, 5, 9, 11 },   // 11507  3 145 P-U-V-L-Y-I-T
  { 0, 1, 2, 4, 6, 10, 11 },  // 14283  1 149 P-U-V-L-N-F-T
  { 0, 1, 2, 4, 6, 9, 11 },   //  8546  3 131 P-U-V-L-N-I-T
  { 0, 1, 2, 5, 6, 10, 11 },  // 11117  3 124 P-U-V-Y-N-F-T
  { 0, 1, 2, 5, 6, 9, 10 },   //  9495  1 112 P-U-V-Y-N-I-F
  { 0, 1, 3, 4, 5, 10, 11 },  //  7590  2  82 P-U-Z-L-Y-F-T
  { 0, 1, 3, 4, 5, 6, 10 },   // 17151  1 163 P-U-Z-L-Y-N-F
  { 0, 1, 3, 4, 5, 6, 11 },   // 10357  2 111 P-U-Z-L-Y-N-T
  { 0, 1, 3, 4, 5, 6, 8 },    //  7693  1 109 P-U-Z-L-Y-N-W
  { 0, 1, 3, 4, 5, 6, 9 },    //  9146  2 115 P-U-Z-L-Y-N-I
  { 0, 1, 4, 5, 6, 10, 11 },  // 19663  4 188 P-U-L-Y-N-F-T
  { 0, 1, 4, 5, 6, 9, 10 },   // 18234  3 227 P-U-L-Y-N-I-F
  { 0, 1, 4, 5, 6, 9, 11 },   // 12198  1 138 P-U-L-Y-N-I-T
  { 0, 2, 3, 4, 5, 6, 10 },   // 18910  4 170 P-V-Z-L-Y-N-F
  { 0, 2, 3, 4, 5, 6, 11 },   // 14417  2 148 P-V-Z-L-Y-N-T
  { 0, 2, 3, 4, 5, 6, 8 },    // 11346  2 163 P-V-Z-L-Y-N-W
  { 0, 2, 4, 5, 6, 10, 11 },  // 18891  3 178 P-V-L-Y-N-F-T
  { 0, 2, 4, 5, 6, 8, 10 },   // 16090  1 240 P-V-L-Y-N-W-F
  { 0, 2, 4, 5, 6, 9, 10 },   // 15587  1 182 P-V-L-Y-N-I-F
  { 0, 2, 4, 5, 8, 10, 11 },  //  7044  1  87 P-V-L-Y-W-F-T
};

void print_orientation(char * prefix, orientation * o) {
  for (int j = 0; j < o->height; j++) {
    printf(prefix);
    for (int i = 0; i < o->width; i++) {
      int idx = j * o->width + i;
      if (o->value[idx]) {
        printf("%c ", o->value[idx]);
      } else {
        printf("  ");
      }
    }
    printf("\n");
  }
}

void print_board(FILE * stream, board * board) {
  fprintf(stream, "flips: ");
  for (int p = 0; p < N_PIECES; p++) {
    if (board->flips[p]) {
      fprintf(stream, "F");
    } else {
      fprintf(stream, "O");
    }
  }
  fprintf(stream, "\n");
  for (int j = 0; j < BOARD_HEIGHT; j++) {
    for (int i = 0; i < BOARD_WIDTH; i++) {
      int idx = j * BOARD_WIDTH + i;
      if (board->cells[idx]) {
        fprintf(stream, "%c ", board->cells[idx]);
      } else {
        fprintf(stream, "  ");
      }
    }
    fprintf(stream, "\n");
  }
}

void initialize_piece(definition * def, piece * piece, bool allow_flip) {
    // printf("Initializing piece\n");
    // print_orientation("", &def->orientation);

    int n_flips = (allow_flip * def->mirror + 1);
    int n_orientations = n_flips * def->rotations;
    piece->n_orientations = n_orientations;

    int value_size = def->orientation.width * def->orientation.height * sizeof(cell);
    int n_values = value_size * n_orientations;
    cell * values = malloc(n_values * sizeof(cell));
    for (int o = 0; o < n_flips; o++) {
      int x_dir = (o == 0 ? 1 : -1);
      for (int r = 0; r < def->rotations; r++) {
        bool shift_i = x_dir * rotations[r].x_x < 0 || x_dir * rotations[r].x_y < 0;
        bool shift_j = rotations[r].y_x < 0 || rotations[r].y_y < 0;

        int idx = o * def->rotations + r;
        cell * value = &values[idx * value_size];
        bool rotated = r % 2 == 0;
        orientation * orientation = &piece->orientations[idx];
        orientation->width = (rotated ? def->orientation.width : def->orientation.height);
        orientation->height = (rotated ? def->orientation.height : def->orientation.width);
        orientation->flipped = o > 0;
        orientation->value = value;
        for (int j = 0; j < orientation->height; j++) {
          for (int i = 0; i < orientation->width; i++) {
            int i_orig = shift_i * (def->orientation.width - 1) + x_dir * (rotations[r].x_x * i + rotations[r].x_y * j);
            int j_orig = shift_j * (def->orientation.height - 1) + rotations[r].y_x * i + rotations[r].y_y * j;
            value[j * orientation->width + i] = def->orientation.value[j_orig * def->orientation.width + i_orig];
          }
        }
        // print_orientation("   ", orientation);
        // printf("\n");
      }
    }
    // printf("\n");
}

board * initialize_board() {
  for (int j = 0; j < BOARD_HEIGHT; j++) {
    for (int i = 0; i < BOARD_WIDTH; i++) {
      initial_board.cells[j * BOARD_WIDTH + i] = 0;
    }
  }
  for (int p = 0; p < N_PIECES; p++) {
    initial_board.flips[p] = false;
  }
  board_set(&initial_board, 6, 0, 'B');
  board_set(&initial_board, 6, 1, 'B');
  board_set(&initial_board, 3, 6, 'B');
  board_set(&initial_board, 4, 6, 'B');
  board_set(&initial_board, 5, 6, 'B');
  board_set(&initial_board, 6, 6, 'B');

  return &initial_board;
}

board * place_piece(board * board, orientation * o, int x, int y) {
  for (int j = 0; j < o->height; j++) {
    for (int i = 0; i < o->width; i++) {
      if (o->value[j * o->width + i] && board_get(board, x + i, y + j)) {
        return 0;
      }
    }
  }
  for (int j = 0; j < o->height; j++) {
    for (int i = 0; i < o->width; i++) {
      int idx = j * o->width + i;
      if (o->value[idx]) {
        board_set(board, x + i, y + j, o->value[idx]);
      }
    }
  }
  return board;
}

board * unplace_piece(board * board, orientation * o, int x, int y) {
  for (int j = 0; j < o->height; j++) {
    for (int i = 0; i < o->width; i++) {
      if (o->value[j * o->width + i]) {
        board_set(board, x + i, y + j, 0);
      }
    }
  }
}

solution * solve(board * board, int p) {
  solution * result = 0;
  if (p == N_PIECES) {
    result = malloc(sizeof(solution));
    result->next = 0;
    memcpy(&result->board, board, sizeof(struct _board));
    // print_board(stdout, board);
    return result;
  }

  for (int o = 0; o < pieces[p].n_orientations; o++) {
    orientation * orientation = &pieces[p].orientations[o];
    for (int y = 0; y <= BOARD_HEIGHT - orientation->height; y++) {
      for (int x = 0; x <= BOARD_WIDTH - orientation->width; x++) {
        struct _board * next_board = place_piece(board, orientation, x, y);
        if (next_board) {
          next_board->flips[p] = orientation->flipped;
          solution * solutions = solve(next_board, p + 1);
          if (solutions) {
            solution ** p_last = &solutions;
            while ((*p_last)->next) {
              p_last = &(*p_last)->next;
            }
            (*p_last)->next = result;
            result = solutions;
          }
          unplace_piece(next_board, orientation, x, y);
          next_board->flips[p] = false;
        }
      }
    }
  }
  return result;
}

void free_solutions(solution * s) {
  if (s == 0) {
    return;
  }
  free(s->next);
}

// (0, 0): 1
// (1, 1): (0, 1) + (1, 0): 2
int binomial(int m, int n) {
  if (m == 0 || n == 0) {
    return 1;
  }
  int result = binomial(m - 1, n) + binomial(m, n - 1);
  return result;
}

void calc_subsets(int m, int n, int* out, int stride) {
  if (n == 0) {
    for (int i = 0; i < m; i++) {
      out[i] = i;
    }
  } else if (m == 0) {
    for (int i = 0; i < n; i++) {
      out[stride - i - 1] = i;
    }
  } else {
    int num_m = binomial(m, n - 1);
    int num_n = binomial(m - 1, n);

    calc_subsets(m, n - 1, out, stride);
    for (int i = 0; i < num_m; i++) {
      out[(i + 1) * stride - n] = m + n - 1;
    }

    int offset = num_m * stride;
    calc_subsets(m - 1, n, &out[offset], stride);
    for (int i = 0; i < num_n; i++) {
      out[offset + i * stride + m - 1] = m + n - 1;
    }
  }
}

char * months[] = {
  "jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"
};
int days[] = {
  31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

void count_solutions(board * board, int subset, int month) {
  // for (int month = 0; month < 12; month++) {
    for (int day = 0; day < days[month]; day++) {
      /*
      char filename[11];
      sprintf(filename, "out/%s-%d", months[month], day + 1);
      printf("%s: ", filename);
      fflush(stdout);
      */
      // printf(" d:%d ", day);
      // fflush(stdout);

      int month_i = month % 6;
      int month_j = month / 6;
      board_set(board, month_i, month_j, 'D');

      int day_i = day % 7;
      int day_j = day / 7 + 2;
      board_set(board, day_i, day_j, 'D');

      solution * solutions = solve(board, 0);
      int n_solutions = 0;
      // FILE * out = fopen(filename, "w");
      while (solutions) {
        n_solutions++;
        // print_board(out, &solutions->board);
        // fprintf(out, "\n");
        solutions = solutions->next;
      }
      // fclose(out);
      printf("%d-%d-%d: %d\n", subset, month, day, n_solutions);
      fflush(stdout);
      free_solutions(solutions);

      board_set(board, day_i, day_j, 0);
      board_set(board, month_i, month_j, 0);

      //if (n_solutions == 0) {
      //  return false;
      //}
    }
  // }
  //return true;
}


int main(int argc, char * argv[]) {
  int test_m = N_PIECES - 1;
  int test_n = N_DEFINITIONS - N_PIECES;
  int n_subsets = binomial(test_m, test_n);
  // printf("(%d, %d): %d\n", test_m, test_n, n_subsets);

  int stride = test_m + test_n;
  int * subsets = malloc(n_subsets * stride * sizeof(int));
  calc_subsets(test_m, test_n, subsets, stride);

  /*
  for (int i = 0; i < n_subsets; i++) {
    for (int j = 0; j < test_m; j++) {
      printf("%d ", buffer[i * stride + j]);
    }
    for (int j = 0; j < test_n; j++) {
      printf("%d ", buffer[i * stride + j + test_m]);
    }
    printf("\n");
  }
  */
  // return 0;

  if (argc < 3) {
    fprintf(stderr, "Usage: %s <id> <total>\n", argv[0]);
    return -1;
  }
  int p_id = atoi(argv[1]);
  int n_p = atoi(argv[2]);
  bool allow_flip = true;

  board * board = initialize_board();
  for (int p = 0; p < N_DEFINITIONS; p++) {
    initialize_piece(&all_definitions[p], &all_pieces[p], allow_flip);
  }

  pieces[0] = all_pieces[0];
  for (int subset = 0; subset < N_COMBINATIONS; subset++) {
    for (int month = 0; month < 12; month++) {
      if ((subset * 12 + month) % n_p != p_id) {
        continue;
      }
      int * combination = combinations[subset];
      for (int piece = 0; piece < N_PIECES - 1; piece++) {
        pieces[piece + 1] = all_pieces[combination[piece] + 1];
      }
      count_solutions(board, subset, month);
    }
    /*
    printf("%d: ", month);
    for (int j = 0; j < test_m; j++) {
      printf("%d", subsets[subset * stride + j]);
    }
    fflush(stdout);
    printf("\n");
    */
  }
  return 0;
}

