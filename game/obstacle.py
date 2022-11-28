import random
import pygame
import os

class Obstacle:
    def __init__(self, position_x, position_y):
        self.position_y = position_y
        self.position_x = position_x
        self.image = None
        self.rect = None

    def draw(self, SCREEN):
        SCREEN.blit(self.image, self.rect)
        #pygame.draw.rect(SCREEN, (0,0,0), pygame.Rect(self.rect.x, self.rect.y, self.rect.width, self.rect.height),  2, 3)

    def update(self, game_speed):
        self.rect.x -= game_speed

class SmallCactus(Obstacle):

    TYPES = [
        pygame.image.load(os.path.join("game/assets/Cactus", "SmallCactus1.png")),
        pygame.image.load(os.path.join("game/assets/Cactus", "SmallCactus2.png")),
        pygame.image.load(os.path.join("game/assets/Cactus", "SmallCactus3.png")),
    ]

    def __init__(self, position_x, position_y):
        super().__init__(position_x, position_y)
        self.image = SmallCactus.TYPES[random.randint(0, 2)]
        self.rect = self.image.get_rect(bottomleft=(self.position_x, self.position_y))

class LargeCactus(Obstacle):

    TYPES = [
        pygame.image.load(os.path.join("game/assets/Cactus", "LargeCactus1.png")),
        pygame.image.load(os.path.join("game/assets/Cactus", "LargeCactus2.png")),
        pygame.image.load(os.path.join("game/assets/Cactus", "LargeCactus3.png")),
    ]

    def __init__(self, position_x, position_y):
        super().__init__(position_x, position_y)
        self.image = LargeCactus.TYPES[random.randint(0, 2)]
        self.rect = self.image.get_rect(bottomleft=(self.position_x, self.position_y))

class Bird(Obstacle):

    TYPES = [
        pygame.image.load(os.path.join("game/assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("game/assets/Bird", "Bird2.png"))
    ]

    def __init__(self, position_x, position_y):
        super().__init__(position_x, position_y)
        self.image = Bird.TYPES[0]
        self.steps = 0
        self.rect = self.image.get_rect(bottomleft=(self.position_x, self.position_y))

    def draw(self, SCREEN):
        if str(self.steps).endswith('1') or str(self.steps).endswith('2') or str(self.steps).endswith('3') or str(self.steps).endswith('4'):
            SCREEN.blit(Bird.TYPES[0], self.rect)
        else:
            SCREEN.blit(Bird.TYPES[1], self.rect)

        self.steps += 1
        #pygame.draw.rect(SCREEN, (0,0,0), pygame.Rect(self.rect.x, self.rect.y, self.rect.width, self.rect.height),  2, 3)


