import numpy as np
import pygame
import random

# Inicializar Pygame
pygame.init()
pygame.display.set_caption("Dino Chrome")

# Definir constantes del juego
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 150
GROUND_HEIGHT = 20
DINO_WIDTH = 44
DINO_HEIGHT = 47
CLOUD_WIDTH = 92
CLOUD_HEIGHT = 27
CACTUS_WIDTH = 17
CACTUS_HEIGHT = 35
CACTUS_MIN_DISTANCE = 150
CACTUS_MAX_DISTANCE = 300
CACTUS_MIN_SPEED = 2
CACTUS_MAX_SPEED = 8
CLOUD_MIN_DISTANCE = 100
CLOUD_MAX_DISTANCE = 200
CLOUD_MIN_SPEED = 1
CLOUD_MAX_SPEED = 3
JUMP_HEIGHT = 80
JUMP_SPEED = 5
GRAVITY = 0.4

# Configuración del agente
GAMMA = 0.9 # Factor de descuento de recompensas futuras
ALPHA = 0.1 # Tasa de aprendizaje de los pesos de la red neuronal
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.005
NUM_EPISODES = 1000

class DinoGame:
    def __init__(self):
        # Inicializar la ventana del juego
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Cargar los recursos del juego
        self.dino_image = pygame.image.load("../game/assets/Dino/DinoRun1.png").convert_alpha()
        self.cloud_image = pygame.image.load("../game/assets/Ground/ground.png").convert_alpha()
        self.cactus_image = pygame.image.load("../game/assets/Cactus/SmallCactus1.png").convert_alpha()
        
        # Inicializar los objetos del juego
        self.dino_position = [50, SCREEN_HEIGHT - GROUND_HEIGHT - DINO_HEIGHT]
        self.jump_speed = 0
        self.cacti = []
        self.clouds = []
        self.score = 0
        self.font = pygame.font.Font(None, 36)
    
    def generate_cactus(self):
        """Generar un nuevo cactus"""
        if len(self.cacti) == 0 or self.cacti[-1][0] < SCREEN_WIDTH - CACTUS_MIN_DISTANCE:
            speed = random.randint(CACTUS_MIN_SPEED, CACTUS_MAX_SPEED)
            distance = random.randint(CACTUS_MIN_DISTANCE, CACTUS_MAX_DISTANCE)
            self.cacti.append([SCREEN_WIDTH + distance, SCREEN_HEIGHT - GROUND_HEIGHT - CACTUS_HEIGHT, speed])
    
    def generate_cloud(self):
        """Generar una nueva nube"""
        if len(self.clouds) == 0 or self.clouds[-1][0] < SCREEN_WIDTH - CLOUD_MIN_DISTANCE:
            speed = random.randint(CLOUD_MIN_SPEED, CLOUD_MAX_SPEED)
            distance = random.randint(CLOUD_MIN_DISTANCE, CLOUD_MAX_DISTANCE)
            self.clouds.append([SCREEN_WIDTH + distance, random.randint(0, SCREEN_HEIGHT // 2), speed])
    
    def move_objects(self):
        """Mover los objetos del juego"""
        # Mover el dinosaurio
        self.dino_position[1] -= self.jump_speed
        self.jump_speed -= GRAVITY
        if self.dino_position[1] >= SCREEN_HEIGHT - GROUND_HEIGHT - DINO_HEIGHT:
            self.dino_position[1] = SCREEN_HEIGHT - GROUND_HEIGHT - DINO_HEIGHT
            self.jump_speed = 0
        
        # Mover los cactus
        for cactus in self.cacti:
            cactus[0] -= cactus[2]
            if cactus[0] < -CACTUS_WIDTH:
                self.cacti.remove(cactus)
                self.score += 1
        
        # Mover las nubes
        for cloud in self.clouds:
            cloud[0] -= cloud[2]
            if cloud[0] < -CLOUD_WIDTH:
                self.clouds.remove(cloud)

    def draw_objects(self):
        """Dibujar los objetos del juego"""
        # Dibujar el fondo
        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (235, 235, 235), pygame.Rect(0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))
        
        # Dibujar el dinosaurio
        self.screen.blit(self.dino_image, self.dino_position)
        
        # Dibujar los cactus
        for cactus in self.cacti:
            self.screen.blit(self.cactus_image, cactus)
        
        # Dibujar las nubes
        for cloud in self.clouds:
            self.screen.blit(self.cloud_image, cloud)
        
        # Dibujar la puntuación
        score_surface = self.font.render(str(self.score), True, (0, 0, 0))
        self.screen.blit(score_surface, (10, 10))

    def is_colliding(self):
        """Comprobar si hay colisión entre el dinosaurio y los cactus"""
        for cactus in self.cacti:
            if self.dino_position[0] + DINO_WIDTH > cactus[0] and self.dino_position[0] < cactus[0] + CACTUS_WIDTH and self.dino_position[1] + DINO_HEIGHT > cactus[1]:
                return True
        return False

    def reset(self):
        """Reiniciar el juego"""
        self.dino_position = [50, SCREEN_HEIGHT - GROUND_HEIGHT - DINO_HEIGHT]
        self.jump_speed = 0
        self.cacti = []
        self.clouds = []
        self.score = 0

    def get_state(self):
        """Obtener el estado actual del juego"""
        state = []
        # Distancia al siguiente cactus
        if len(self.cacti) > 0:
            state.append(self.cacti[0][0] - self.dino_position[0])
        else:
            state.append(SCREEN_WIDTH - self.dino_position[0])
        # Distancia a la siguiente nube
        if len(self.clouds) > 0:
            state.append(self.clouds[0][0] - self.dino_position[0])
        else:
            state.append(SCREEN_WIDTH - self.dino_position[0])
        # Altura del dinosaurio
        state.append(self.dino_position[1])
        return np.array(state)

    def step(self, action):
        """Ejecutar una acción en el juego"""
        if action == 0 and self.dino_position[1] == SCREEN_HEIGHT - GROUND_HEIGHT - DINO_HEIGHT:
            self.jump_speed = JUMP_SPEED
        
        self.generate_cactus()
        self.generate_cloud()
        self.move_objects()
        
        if self.is_colliding():
            reward = -100
            done = True
            self.reset()
        else:
            reward = 1
            done = False
        
        return self.get_state(), reward, done
    
def update_q(q_table, state, action, reward, next_state):
    q_old = q_table[tuple(state)][action]
    q_new = reward + GAMMA * np.max(q_table[tuple(next_state)])
    q_table[tuple(state)][action] = (1 - ALPHA) * q_old + ALPHA * q_new
    return q_table

    
# Inicializar el juego
pygame.init()
game = DinoGame()

print("a")

# Inicializar la tabla Q
q_table = {}
for i in range(-300, SCREEN_WIDTH + 1):
    for j in range(-300, SCREEN_WIDTH + 1):
        for k in range(0, SCREEN_HEIGHT + 1):
            q_table[(i, j, k)] = [0, 0]

print("aa")

# Entrenar el agente
epsilon = 1
for episode in range(NUM_EPISODES):
    state = game.get_state()
    done = False
    while not done:
        # Selección de acción
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(q_table[tuple(state)])
        # Ejecutar acción
        next_state, reward, done = game.step(action)
        # Actualizar tabla Q
        q_table = update_q(q_table, state, action, reward, next_state)
        # Actualizar estado
        state = next_state
    # Reducir el valor de epsilon gradualmente
    epsilon = max(0.1, epsilon - 0.005)
    # Imprimir la puntuación del episodio
    print('Episode {}: Score = {}'.format(episode, game.score))

# Jugar con el agente entrenado
game.reset()
done = False
while not done:
    state = game.get_state()
    action = np.argmax(q_table[tuple(state)])
    next_state, reward, done = game.step(action)
    game.draw_objects()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break

pygame.quit()

