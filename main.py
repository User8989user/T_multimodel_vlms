
# ================================================================
# 1. Установка зависимостей и клонирование nanoVLM
# ================================================================
!pip install -q minigrid gymnasium torch torchvision transformers matplotlib numpy pandas tqdm wandb

# Клонируем репозиторий nanoVLM (содержит предобученные веса и код модели) Если мы в Колаб
""" !git clone https://github.com/huggingface/nanoVLM.git
# %cd nanoVLM
!pip install -e .  # установка в режиме разработки
# %cd .. """

import sys
sys.path.append('/content/nanoVLM')

import os
import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import random
from collections import deque
import wandb
from PIL import Image

# Импортируем компоненты nanoVLM
from nanovlm import NanoVLMForConditionalGeneration, NanoVLMProcessor
# Для классификации мы будем использовать только vision tower
from nanovlm.modeling_nanovlm import NanoVLMModel
from transformers import AutoConfig, AutoModel

# Установим seed для воспроизводимости
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Определим устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================================================
# 2. Среда MiniGrid EmptyEnv и создание эксперта
# ================================================================
ENV_NAME = "MiniGrid-Empty-8x8-v0"  # можно также 5x5
MAX_STEPS = 100

def make_env(env_name=ENV_NAME, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    return env

# Эксперт на основе A* (использует внутреннюю карту среды)
class AStarExpert:
    def __init__(self, env):
        self.env = env
        self.dir_vectors = [(1,0), (0,1), (-1,0), (0,-1)]  # right, down, left, up

    def get_action(self):
        # Получаем позицию и направление агента из внутреннего состояния среды
        agent_pos = self.env.agent_pos
        agent_dir = self.env.agent_dir

        # Находим цель (Goal)
        goal_pos = None
        grid = self.env.grid
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell and cell.type == 'goal':
                    goal_pos = (i, j)
                    break
        if goal_pos is None:
            return 2  # forward (заглушка)

        # Построим путь A* (манхэттенское расстояние, проходимы все клетки кроме стен)
        path = self.a_star(agent_pos, goal_pos, grid)
        if len(path) < 2:
            return 2  # уже на цели? не должно быть

        next_cell = path[1]
        # Определяем желаемое направление
        desired_dir_vec = (next_cell[0] - agent_pos[0], next_cell[1] - agent_pos[1])
        current_dir_vec = self.dir_vectors[agent_dir]

        if desired_dir_vec == current_dir_vec:
            return 2  # forward
        # Проверяем, надо ли повернуть налево или направо
        left_dir_vec = self.dir_vectors[(agent_dir - 1) % 4]
        if desired_dir_vec == left_dir_vec:
            return 0  # left
        else:
            return 1  # right

    def a_star(self, start, goal, grid):
        # Простой A* для сетки без препятствий (в EmptyEnv только стены по краям)
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: abs(start[0]-goal[0]) + abs(start[1]-goal[1])}

        while open_set:
            current = min(open_set, key=lambda x: x[0])[1]
            open_set = [item for item in open_set if item[1] != current]

            if current == goal:
                # Восстанавливаем путь
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                neighbor = (current[0]+dx, current[1]+dy)
                # Проверяем границы и проходимость
                if (0 <= neighbor[0] < grid.width and 0 <= neighbor[1] < grid.height):
                    cell = grid.get(neighbor[0], neighbor[1])
                    # Если клетка стена, пропускаем
                    if cell and cell.type == 'wall':
                        continue
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1])
                        f_score[neighbor] = f
                        open_set.append((f, neighbor))
        return [start]  # путь не найден (не должно быть)

# Функция для сбора экспертных траекторий
def collect_expert_trajectories(env, expert, num_episodes=500, max_steps=MAX_STEPS):
    data_images = []
    data_actions = []
    success_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        expert.env = env  # обновляем ссылку на среду для эксперта
        trajectory_images = []
        trajectory_actions = []
        for step in range(max_steps):
            action = expert.get_action()
            # Сохраняем наблюдение (изображение) и действие
            img = obs['image']  # (7,7,3)
            trajectory_images.append(img.copy())
            trajectory_actions.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                if reward > 0:
                    success_count += 1
                break

        # Добавляем всю траекторию в общий датасет
        data_images.extend(trajectory_images)
        data_actions.extend(trajectory_actions)

    print(f"Эксперт завершил {num_episodes} эпизодов, успешных: {success_count} ({success_count/num_episodes:.2%})")
    print(f"Всего собрано переходов: {len(data_images)}")
    return np.array(data_images), np.array(data_actions)

# Создаём среду и эксперта, собираем данные
env = make_env()
expert = AStarExpert(env)
images, actions = collect_expert_trajectories(env, expert, num_episodes=200)  # для быстрого теста 200 эпизодов
print(f"Размер датасета: {images.shape}, {actions.shape}")

# ================================================================
# 3. Подготовка модели для классификации действий (прямой вывод)
# ================================================================
# Загрузим предобученный Vision Transformer из nanoVLM
# В nanoVLM используется модель CLIP ViT, возьмём её отдельно

from transformers import CLIPVisionModel, CLIPImageProcessor

class VisionClassifier(nn.Module):
    def __init__(self, num_actions=7, vision_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        hidden_size = self.vision_encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_actions)

    def forward(self, pixel_values):
        # pixel_values: (B, C, H, W)
        outputs = self.vision_encoder(pixel_values)
        # используем [CLS] токен
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        logits = self.classifier(cls_embedding)
        return logits

    def get_action(self, obs_image, device='cuda'):
        # obs_image: numpy array (H, W, C) в диапазоне 0-255
        # преобразуем в тензор
        if isinstance(obs_image, np.ndarray):
            # Normalize и resize до нужного размера (224x224)
            inputs = self.image_processor(images=obs_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
        else:
            pixel_values = obs_image.to(device)
        with torch.no_grad():
            logits = self.forward(pixel_values)
            action = torch.argmax(logits, dim=-1).item()
        return action

model = VisionClassifier(num_actions=7).to(device)
print(model)

# ================================================================
# 4. Датасет и DataLoader для SFT
# ================================================================
class MinigridDataset(Dataset):
    def __init__(self, images, actions, image_processor):
        self.images = images  # numpy array (N, H, W, C) uint8
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.image_processor = image_processor

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Применяем обработку (resize, normalize)
        inputs = self.image_processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # убираем batch
        action = self.actions[idx]
        return pixel_values, action

# Создаём датасет
dataset = MinigridDataset(images, actions, model.image_processor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Разделим на train/val (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ================================================================
# 5. SFT обучение
# ================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 5
train_losses = []
val_accs = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (pixel_values, actions) in enumerate(train_loader):
        pixel_values = pixel_values.to(device)
        actions = actions.to(device)

        logits = model(pixel_values)
        loss = criterion(logits, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Валидация
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for pixel_values, actions in val_loader:
            pixel_values = pixel_values.to(device)
            actions = actions.to(device)
            logits = model(pixel_values)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == actions).sum().item()
            total += actions.size(0)
    val_acc = correct / total
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

# Построим графики обучения
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ================================================================
# 6. Оценка SFT политики в среде
# ================================================================
def evaluate_policy(env, model, num_episodes=50, max_steps=MAX_STEPS, deterministic=True, device='cuda'):
    success_count = 0
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_return = 0
        for step in range(max_steps):
            action = model.get_action(obs['image'], device)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            if terminated or truncated:
                if reward > 0:
                    success_count += 1
                break
        returns.append(ep_return)
    success_rate = success_count / num_episodes
    avg_return = np.mean(returns)
    return success_rate, avg_return

# Оценим дообученную модель
sft_success, sft_return = evaluate_policy(env, model, num_episodes=50)
print(f"SFT: Success rate = {sft_success:.2%}, Avg return = {sft_return:.4f}")

# ================================================================
# 7. GRPO (упрощённо: PPO с KL-регуляризацией к SFT модели)
# ================================================================
# Будем использовать ту же архитектуру классификатора.
# Референсная модель (замороженная) – это наша обученная SFT модель.
# Обучаемая модель инициализируется её весами.

class PolicyNetwork(VisionClassifier):
    def __init__(self, num_actions, vision_model_name):
        super().__init__(num_actions, vision_model_name)

    def get_action_probs(self, pixel_values):
        logits = self.forward(pixel_values)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_log_prob(self, pixel_values, actions):
        logits = self.forward(pixel_values)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

# Создаём политику и референсную модель
policy = PolicyNetwork(num_actions=7, vision_model_name="openai/clip-vit-base-patch32").to(device)
policy.load_state_dict(model.state_dict())  # инициализируем от SFT

ref_policy = PolicyNetwork(num_actions=7, vision_model_name="openai/clip-vit-base-patch32").to(device)
ref_policy.load_state_dict(model.state_dict())
ref_policy.eval()
for param in ref_policy.parameters():
    param.requires_grad = False

# Параметры PPO
lr = 3e-5
gamma = 0.99
lam = 0.95
clip_epsilon = 0.2
kl_coef = 0.01  # коэффициент KL штрафа
ppo_epochs = 4
batch_size = 64
rollout_steps = 2048  # количество шагов сбора данных за итерацию

optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

# Функция для сбора роллаутов
def collect_rollouts(env, policy, num_steps=rollout_steps):
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    log_prob_list = []
    value_list = []  # для PPO нужна оценка ценности, но у нас нет критика. Упростим: будем использовать return-to-go как advantage.
    # Поскольку задача простая, можно обойтись без нейросетевого критика, используя монте-карловские returns.
    # Но для простоты реализуем сбор траекторий и затем вычислим returns.

    obs, _ = env.reset()
    for _ in range(num_steps):
        # Преобразуем наблюдение
        img = obs['image']
        inputs = policy.image_processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)

        with torch.no_grad():
            logits = policy(pixel_values)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Выполняем действие
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_list.append(img)
        action_list.append(action.item())
        reward_list.append(reward)
        done_list.append(done)
        log_prob_list.append(log_prob.item())

        if done:
            obs, _ = env.reset()

    # Вычисляем returns (discounted)
    returns = []
    G = 0
    for r, done in zip(reversed(reward_list), reversed(done_list)):
        if done:
            G = 0
        G = r + gamma * G
        returns.insert(0, G)

    # Преобразуем в тензоры
    obs_tensors = []
    for img in obs_list:
        inputs = policy.image_processor(images=img, return_tensors="pt")
        obs_tensors.append(inputs['pixel_values'].squeeze(0))
    obs_tensors = torch.stack(obs_tensors).to(device)
    actions_tensor = torch.tensor(action_list, device=device)
    old_log_probs = torch.tensor(log_prob_list, device=device)
    returns_tensor = torch.tensor(returns, device=device)

    return obs_tensors, actions_tensor, old_log_probs, returns_tensor

# Основной цикл PPO
ppo_iterations = 50
success_rates_ppo = []
returns_ppo = []

for iteration in range(ppo_iterations):
    # Сбор роллаутов
    obs_batch, actions_batch, old_log_probs_batch, returns_batch = collect_rollouts(env, policy)

    # Нормализация returns (опционально)
    returns_batch = (returns_batch - returns_batch.mean()) / (returns_batch.std() + 1e-8)

    # Оптимизация PPO
    for _ in range(ppo_epochs):
        # Берём мини-батчи
        perm = torch.randperm(len(obs_batch))
        for i in range(0, len(obs_batch), batch_size):
            idx = perm[i:i+batch_size]
            obs_mb = obs_batch[idx]
            actions_mb = actions_batch[idx]
            old_log_probs_mb = old_log_probs_batch[idx]
            returns_mb = returns_batch[idx]

            # Текущие логиты и log probs
            logits = policy(obs_mb)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions_mb)
            entropy = dist.entropy().mean()

            # Отношение вероятностей
            ratio = torch.exp(log_probs - old_log_probs_mb)
            # PPO clip objective
            surr1 = ratio * returns_mb
            surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * returns_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL divergence с референсной политикой (для guided regularization)
            with torch.no_grad():
                ref_logits = ref_policy(obs_mb)
                ref_probs = F.softmax(ref_logits, dim=-1)
            kl = (ref_probs * (torch.log(ref_probs + 1e-8) - torch.log(probs + 1e-8))).sum(-1).mean()

            # Общая потеря
            loss = policy_loss + kl_coef * kl - 0.01 * entropy  # энтропийный бонус

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Оценка после итерации
    success, avg_return = evaluate_policy(env, policy, num_episodes=20)
    success_rates_ppo.append(success)
    returns_ppo.append(avg_return)
    print(f"Iter {iteration+1}: success rate = {success:.2%}, return = {avg_return:.4f}")

# Сравнительный график
plt.figure(figsize=(10,5))
plt.plot(success_rates_ppo, label='PPO (GRPO) success rate')
plt.axhline(y=sft_success, color='r', linestyle='--', label='SFT baseline')
plt.xlabel('PPO Iteration')
plt.ylabel('Success Rate')
plt.legend()
plt.title('PPO vs SFT')
plt.show()

# ================================================================
# 8. GRPO с текстом+действие (концептуально)
# ================================================================
# Для этого варианта нам нужно использовать генеративную модель nanoVLM.
# Мы загружаем полную модель, которая умеет генерировать текст по изображению и промпту.
# Затем мы определяем формат вывода, например:
# "I see the goal ahead. I move forward. Action: forward"
# Действие извлекаем по ключевому слову "Action:".

# Из-за сложности реализации PPO с генерацией текста, здесь представлен только каркас.
# Основные шаги:
# 1. Загрузить NanoVLMForConditionalGeneration и процессор.
# 2. Подготовить SFT датасет с текстовыми ответами (эксперт должен генерировать текст).
# 3. Обучить SFT на генерацию текста (аналогично п.3, но с loss на токены).
# 4. Для RL будем использовать политику, которая генерирует текст, а действие извлекается.
#    Вероятность действия можно оценить как сумму логов токенов, соответствующих действию (если оно представлено несколькими токенами).
#    Для упрощения можно считать, что действие всегда представлено одним токеном (например, "forward").
# 5. Референсная модель (замороженная) для KL-регуляризации.
# 6. PPO обновление: цель – максимизировать преимущество, регуляризация KL между распределениями токенов.
#
# Ниже приведён пример загрузки модели и токенизатора, а также функция извлечения действия.

from transformers import AutoTokenizer, AutoModelForCausalLM
from nanovlm import NanoVLMProcessor

# Загружаем предобученную nanoVLM (например, "nanoVLM-160M")
# model_name = "huggingface/nanoVLM-160M"  # такого нет в официальном хабе, поэтому используем заглушку
# В реальности нужно будет скачать веса из репозитория nanoVLM, но для демо просто покажем идею.

# processor = NanoVLMProcessor.from_pretrained("path/to/nanovlm")
# tokenizer = processor.tokenizer
# image_processor = processor.image_processor
# model = NanoVLMForConditionalGeneration.from_pretrained("path/to/nanovlm").to(device)

# Пример извлечения действия из текста:
def extract_action_from_text(text, action_map={'left':0, 'right':1, 'forward':2}):
    # Ищем подстроку "Action: " и берём следующее слово
    import re
    match = re.search(r'Action:\s*(\w+)', text)
    if match:
        action_word = match.group(1)
        return action_map.get(action_word, None)
    return None

# Для обучения RL потребуется генерировать текст, получать награду из среды,
# вычислять log-вероятности сгенерированных токенов и обновлять модель.
# Это достаточно сложно и выходит за рамки данного примера.
# В качестве демонстрации можно запустить предобученную модель и оценить её в среде,
# но без RL-дообучения.

print("Вариант с текстом+действие требует отдельной реализации, основанной на генеративных моделях.")
print("Здесь представлена только концепция.")

# ================================================================
# 9. Итоговое сравнение (таблица и выводы)
# ================================================================
# Сведём результаты в таблицу
print("\n--- Результаты ---")
print(f"SFT: Success Rate = {sft_success:.2%}, Avg Return = {sft_return:.4f}")
print(f"PPO (GRPO-action): Final Success Rate = {success_rates_ppo[-1]:.2%}, Final Avg Return = {returns_ppo[-1]:.4f}")
print(f"Для текст+действие не удалось получить численных результатов в данной демо-версии.")

# Построим сводный график
plt.figure(figsize=(8,5))
plt.plot(success_rates_ppo, label='GRPO-action (PPO)')
plt.axhline(y=sft_success, color='r', linestyle='--', label='SFT baseline')
plt.xlabel('PPO Iteration')
plt.ylabel('Success Rate')
plt.title('Сравнение SFT и GRPO-action')
plt.legend()
plt.grid(True)
plt.show()

# ================================================================
# 10. Сохранение модели
# ================================================================
# Сохраним веса обученной политики
torch.save(policy.state_dict(), 'policy_ppo.pth')
print("Модель сохранена как policy_ppo.pth")