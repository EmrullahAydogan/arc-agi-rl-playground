#!/usr/bin/env python3
"""
Sistem test scripti (GUI olmadan)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import ARCDataLoader
from src.environment.arc_env import ARCEnvironment
from src.agents.random_agent import RandomAgent


def test_data_loader():
    print("=" * 60)
    print("📦 Data Loader Test")
    print("=" * 60)

    loader = ARCDataLoader("arc-prize-2025")

    # Training data yükle
    print("🔄 Training data yükleniyor...")
    challenges, solutions = loader.load_training_data()
    print(f"✅ {len(challenges)} training puzzle yüklendi")

    # Rastgele puzzle al
    puzzle_id, puzzle_data = loader.get_random_puzzle()
    print(f"🎲 Rastgele puzzle: {puzzle_id}")
    print(f"   Train examples: {len(puzzle_data['train'])}")
    print(f"   Test examples: {len(puzzle_data['test'])}")

    return puzzle_data


def test_environment(puzzle_data):
    print("\n" + "=" * 60)
    print("🌍 Environment Test")
    print("=" * 60)

    # Environment oluştur
    env = ARCEnvironment(puzzle_data, task_index=0, max_steps=50)
    print(f"✅ Environment oluşturuldu")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space.shape}")

    # Reset
    observation, info = env.reset()
    print(f"🔄 Environment reset edildi")
    print(f"   Grid shape: {info['puzzle_shape']}")
    print(f"   Input grid:\n{info['input_grid']}")
    print(f"   Target grid:\n{info['target_grid']}")

    return env


def test_agent_environment_loop(env):
    print("\n" + "=" * 60)
    print("🤖 Agent-Environment Loop Test")
    print("=" * 60)

    # Agent oluştur
    agent = RandomAgent(env.action_space, env.observation_space, seed=42)
    print(f"✅ {agent.__class__.__name__} oluşturuldu")

    # Reset
    observation, info = env.reset()
    agent.reset()

    # Birkaç adım çalıştır
    print(f"\n🏃 10 adım çalıştırılıyor...")
    for step in range(10):
        # Action seç
        action = agent.select_action(observation, info)

        # Step
        next_observation, reward, terminated, truncated, step_info = env.step(action)

        # Güncelle
        agent.update(observation, action, reward, next_observation, terminated or truncated, step_info)

        # Bilgi göster
        x, y, color = env._action_to_grid_operation(action)
        print(f"   Step {step + 1}: Action=({x},{y},{color}), Reward={reward:.2f}, Done={terminated or truncated}")

        if terminated or truncated:
            print(f"\n{'🎉 ÇÖZÜLDÜ!' if terminated else '⏱️  Zaman aşımı'}")
            break

        observation = next_observation

    # Final state
    state_info = env.get_state_info()
    print(f"\n📊 Final Stats:")
    print(f"   Steps: {state_info['steps']}")
    print(f"   Total Reward: {state_info['total_reward']:.2f}")
    print(f"   Solved: {state_info['is_solved']}")
    print(f"   Agent Actions: {agent.get_info()['actions_taken']}")


def main():
    print("\n" + "=" * 60)
    print("🧪 ARC-AGI RL SYSTEM TEST")
    print("=" * 60)
    print()

    try:
        # Test 1: Data Loader
        puzzle_data = test_data_loader()

        # Test 2: Environment
        env = test_environment(puzzle_data)

        # Test 3: Agent-Environment Loop
        test_agent_environment_loop(env)

        print("\n" + "=" * 60)
        print("✅ TÜM TESTLER BAŞARILI!")
        print("=" * 60)
        print("\n💡 Sistemi çalıştırmak için:")
        print("   python playground.py")
        print()

    except Exception as e:
        print(f"\n❌ TEST BAŞARISIZ: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
