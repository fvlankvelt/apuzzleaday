#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BOARD_WIDTH 7
#define BOARD_HEIGHT 7

#define N_PIECES 8

typedef unsigned char bool;
#define true ((bool) 1)
#define false ((bool) 0)
#define X ((bool) 1)
#define O ((bool) 0)

typedef struct _orientation {
  int height;
  int width;
  bool * value;
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

typedef struct _rotation {
  int x_x;
  int x_y;
  int y_x;
  int y_y;
} rotation;

typedef struct _frame {
  int x;
  int y;
  orientation * orientation;
} frame;

typedef struct _solution {
  frame stack[N_PIECES];
  struct _solution * next;
} solution;

bool initial_board[BOARD_WIDTH * BOARD_HEIGHT];

static inline void board_set(bool * board, int i, int j, bool value) {
  board[j * BOARD_WIDTH + i] = value;
}

static inline bool board_get(bool * board, int i, int j) {
  return board[j * BOARD_WIDTH + i];
}

definition definitions[N_PIECES] = {
  { false, 2,
    { 2, 3, (bool[]) {
      X, X, X,
      X, X, X } } },
  { true, 4,
    { 2, 3, (bool[]) {
      X, X, X,
      X, X, O } } },
  { false, 4,
    { 2, 3, (bool[]) {
      X, X, X,
      X, O, X } } },
  { false, 4,
    { 3, 3, (bool[]) {
      X, X, X,
      X, O, O,
      X, O, O, } } },
  { true, 2,
    { 3, 3, (bool[]) {
      X, X, O,
      O, X, O,
      O, X, X, } } },
  { true, 4,
    { 2, 4, (bool[]) {
      X, X, X, X,
      X, O, O, O } } },
  { true, 4,
    { 2, 4, (bool[]) {
      X, X, X, X,
      O, X, O, O } } },
  { true, 4,
    { 2, 4, (bool[]) {
      O, X, X, X,
      X, X, O, O } } },
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
			if (o->value[j * o->width + i]) {
				printf("x ");
			} else {
				printf("  ");
			}
		}
		printf("\n");
	}
}

void print_board(bool * board) {
  for (int j = 0; j < BOARD_HEIGHT; j++) {
    for (int i = 0; i < BOARD_WIDTH; i++) {
      if (board[j * BOARD_WIDTH + i]) {
        printf("x ");
      } else {
        printf("  ");
      }
    }
    printf("\n");
  }
}

void initialize_piece(definition * def, piece * piece) {
    // printf("Initializing piece\n");
    // print_orientation("", &def->orientation);

    int n_orientations = (def->mirror + 1) * def->rotations;
    piece->n_orientations = n_orientations;

    int value_size = def->orientation.width * def->orientation.height * sizeof(bool);
    int n_values = value_size * n_orientations;
    bool * values = malloc(n_values * sizeof(bool));
    for (int o = 0; o <= def->mirror; o++) {
      int x_dir = (o == 0 ? 1 : -1);
      for (int r = 0; r < def->rotations; r++) {
        bool shift_i = x_dir * rotations[r].x_x < 0 || x_dir * rotations[r].x_y < 0;
        bool shift_j = rotations[r].y_x < 0 || rotations[r].y_y < 0;

        int idx = o * def->rotations + r;
        bool * value = &values[idx * value_size];
        bool rotated = r % 2 == 0;
        orientation * orientation = &piece->orientations[idx];
        orientation->width = (rotated ? def->orientation.width : def->orientation.height);
        orientation->height = (rotated ? def->orientation.height : def->orientation.width);
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

bool * initialize_board() {
  for (int j = 0; j < BOARD_HEIGHT; j++) {
    for (int i = 0; i < BOARD_WIDTH; i++) {
      initial_board[j * BOARD_WIDTH + i] = false;
    }
  }
  board_set(initial_board, 6, 0, true);
  board_set(initial_board, 6, 1, true);
  board_set(initial_board, 3, 6, true);
  board_set(initial_board, 4, 6, true);
  board_set(initial_board, 5, 6, true);
  board_set(initial_board, 6, 6, true);

  return initial_board;
}

bool * place_piece(bool * board, orientation * o, int x, int y) {
  for (int j = 0; j < o->height; j++) {
    for (int i = 0; i < o->width; i++) {
      if (o->value[j * o->width + i] && board[(y + j) * BOARD_WIDTH + (x + i)]) {
        return 0;
      }
    }
  }
  for (int j = 0; j < o->height; j++) {
    for (int i = 0; i < o->width; i++) {
      if (o->value[j * o->width + i]) {
        board[(y + j) * BOARD_WIDTH + (x + i)] = true;
      }
    }
  }
  return board;
}

bool * unplace_piece(bool * board, orientation * o, int x, int y) {
  for (int j = 0; j < o->height; j++) {
    for (int i = 0; i < o->width; i++) {
      if (o->value[j * o->width + i]) {
        board[(y + j) * BOARD_WIDTH + (x + i)] = false;
      }
    }
  }
}

solution * solve(bool * board, int p, frame * stack) {
  solution * result = 0;
  if (p == N_PIECES) {
    print_board(board);
    result = malloc(sizeof(solution));
    result->next = 0;
    memcpy(result->stack, stack, p * sizeof(frame));
    return result;
  }

  solution ** p_last = 0;
  frame * next_stack = malloc((p + 1) * sizeof(frame));
  memcpy(next_stack, stack, p * sizeof(frame));
  for (int o = 0; o < pieces[p].n_orientations; o++) {
    orientation * orientation = &pieces[p].orientations[o];
    for (int y = 0; y <= BOARD_HEIGHT - orientation->height; y++) {
      for (int x = 0; x <= BOARD_WIDTH - orientation->width; x++) {
        bool * next_board = place_piece(board, orientation, x, y);
        if (next_board) {
          solution * solutions = solve(next_board, p + 1, next_stack);
          if (solutions) {
            solution ** p_last = &solutions;
            while ((*p_last)->next) {
              p_last = &(*p_last)->next;
            }
            (*p_last)->next = result;
            result = solutions;
          }
          unplace_piece(next_board, orientation, x, y);
        }
      }
    }
  }
  return result;
}

int main() {
  bool * board = initialize_board();
  board_set(board, 1, 1, true);
  board_set(board, 6, 3, true);
	for (int p = 0; p < N_PIECES; p++) {
    initialize_piece(&definitions[p], &pieces[p]);
	}
  solution * solutions = solve(board, 0, 0);
}
