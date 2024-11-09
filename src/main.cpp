#include "raylib.h"
#include "game.h"
#include <iostream>
#include <random>

struct Time
{
  double lastUpdateTime = 0;

  bool eventTriggered(double interval)
  {
    double currentTime = GetTime();
    if (currentTime - lastUpdateTime >= interval)
    {
      lastUpdateTime = currentTime;
      return true;
    }
    return false;
  }
};

int main()
{
  InitWindow(500, 600, "Tetris");
  SetTargetFPS(70);

  Game game = Game();
  Time t;

  Font font = LoadFontEx("fonts/press_start.ttf", 64, 0, 0);

  while (!WindowShouldClose())
  {
    game.handleInput();

    if (t.eventTriggered(0.2))
      game.moveBlockDown();

    BeginDrawing();

    ClearBackground(WHITE);

    DrawRectangle(300, 0, GetScreenWidth(), GetScreenHeight(), BLACK);

    DrawTextEx(font, "Score", {350, 15}, 38, 2, WHITE);
    DrawRectangleRounded({320, 55, 170, 60}, 0.3, 6, WHITE);

    // Score
    char scoreText[10];
    sprintf(scoreText, "%d", game.score);
    Vector2 textSize = MeasureTextEx(font, scoreText, 38, 2);
    DrawTextEx(font, scoreText, {320 + (170 - textSize.x)/2, 65}, 38, 2, BLACK);

    DrawTextEx(font, "Next Block", {310, 175}, 36, 2, WHITE);
    DrawRectangleRounded({320, 215, 170, 180}, 0.3, 6, WHITE);


    if (game.gameOver) DrawTextEx(font, "GAME OVER", {307, 450}, 32, 2, WHITE);

    Rectangle restartBounds = Rectangle({320, 540, 170, 40});
    DrawRectangleRounded(restartBounds, 0.6, 6, RED);
    DrawTextEx(font, "RESTART", {333, 545}, 30, 2, WHITE);

    if (CheckCollisionPointRec(GetMousePosition(), restartBounds)){
      if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)){
        game.reset();
      }
    }

    game.draw();

    EndDrawing();
  }

  CloseWindow();
}