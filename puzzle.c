#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BOARD_WIDTH 7
#define BOARD_HEIGHT 7

#define N_PIECES 8

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

definition definitions[N_PIECES] = {
  { false, 2,
    { 2, 3, false, (cell[]) {
      'O', 'O', 'O',
      'O', 'O', 'O' } } },
  { true, 4,
    { 2, 3, false, (cell[]) {
      'P', 'P',  0,
      'P', 'P', 'P' } } },
  { false, 4,
    { 2, 3, false, (cell[]) {
      'C', 'C', 'C',
      'C',   0, 'C' } } },
  { false, 4,
    { 3, 3, false, (cell[]) {
      'L', 'L', 'L',
      'L',   0,   0,
      'L',   0,   0, } } },
  { true, 2,
    { 3, 3, false, (cell[]) {
      'S', 'S',   0,
        0, 'S',   0,
        0, 'S', 'S', } } },
  { true, 4,
    { 2, 4, false, (cell[]) {
      'j', 'j', 'j', 'j',
      'j',   0,   0,   0 } } },
  { true, 4,
    { 2, 4, false, (cell[]) {
      'T', 'T', 'T', 'T',
        0,   0, 'T',   0 } } },
  { true, 4,
    { 2, 4, false, (cell[]) {
        0, 'Z', 'Z', 'Z',
      'Z', 'Z',   0,   0 } } },
};

piece pieces[N_PIECES];

rotation rotations[4] = {
  {  1,  0,  0,  1 },
  {  0,  1, -1,  0 },
  { -1,  0,  0, -1 },
  {  0, -1,  1,  0 },
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

char * months[] = {
  "jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"
};
int days[] = {
  31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

int main(int argc, char * argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <month> [--noflip]\n", argv[0]);
    return -1;
  }
  int month = atoi(argv[1]);
  bool allow_flip = (argc < 3) || (strcmp(argv[2], "--noflip") != 0);

  board * board = initialize_board();
	for (int p = 0; p < N_PIECES; p++) {
    initialize_piece(&definitions[p], &pieces[p], allow_flip);
	}

  // for (int month = 0; month < 12; month++) {
    for (int day = 0; day < days[month]; day++) {
      char filename[11];
      sprintf(filename, "out/%s-%d", months[month], day + 1);
      printf("%s: ", filename);
      fflush(stdout);

      int month_i = month % 6;
      int month_j = month / 6;
      board_set(board, month_i, month_j, 'D');

      int day_i = day % 7;
      int day_j = day / 7 + 2;
      board_set(board, day_i, day_j, 'D');

      solution * solutions = solve(board, 0);
      int n_solutions = 0;
      FILE * out = fopen(filename, "w");
      while (solutions) {
        n_solutions++;
        print_board(out, &solutions->board);
        fprintf(out, "\n");
        solutions = solutions->next;
      }
      fclose(out);
      printf("%d\n", n_solutions);

      board_set(board, day_i, day_j, 0);
      board_set(board, month_i, month_j, 0);
    }
  // }
}
