"""
ARC-AGI veri yükleme ve yönetme modülü
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random


class ARCDataLoader:
    """ARC-AGI puzzle verilerini yükler ve yönetir"""

    def __init__(self, data_dir: str = "arc-prize-2025"):
        """
        Args:
            data_dir: ARC veri dosyalarının bulunduğu dizin
        """
        self.data_dir = Path(data_dir)
        self.training_challenges = None
        self.training_solutions = None
        self.evaluation_challenges = None
        self.evaluation_solutions = None

    def load_training_data(self) -> Tuple[Dict, Dict]:
        """Training verilerini yükle"""
        if self.training_challenges is None:
            challenges_path = self.data_dir / "arc-agi_training_challenges.json"
            solutions_path = self.data_dir / "arc-agi_training_solutions.json"

            with open(challenges_path, 'r') as f:
                self.training_challenges = json.load(f)
            with open(solutions_path, 'r') as f:
                self.training_solutions = json.load(f)

        return self.training_challenges, self.training_solutions

    def load_evaluation_data(self) -> Tuple[Dict, Dict]:
        """Evaluation verilerini yükle"""
        if self.evaluation_challenges is None:
            challenges_path = self.data_dir / "arc-agi_evaluation_challenges.json"
            solutions_path = self.data_dir / "arc-agi_evaluation_solutions.json"

            with open(challenges_path, 'r') as f:
                self.evaluation_challenges = json.load(f)
            with open(solutions_path, 'r') as f:
                self.evaluation_solutions = json.load(f)

        return self.evaluation_challenges, self.evaluation_solutions

    def get_puzzle(self, puzzle_id: str, dataset: str = "training") -> Dict:
        """
        Belirli bir puzzle'ı al

        Args:
            puzzle_id: Puzzle ID'si
            dataset: "training" veya "evaluation"

        Returns:
            Puzzle verisi (train ve test içeren dict)
        """
        if dataset == "training":
            challenges, _ = self.load_training_data()
        else:
            challenges, _ = self.load_evaluation_data()

        return challenges.get(puzzle_id)

    def get_random_puzzle(self, dataset: str = "training") -> Tuple[str, Dict]:
        """
        Rastgele bir puzzle al

        Args:
            dataset: "training" veya "evaluation"

        Returns:
            (puzzle_id, puzzle_data) tuple'ı
        """
        if dataset == "training":
            challenges, _ = self.load_training_data()
        else:
            challenges, _ = self.load_evaluation_data()

        puzzle_id = random.choice(list(challenges.keys()))
        return puzzle_id, challenges[puzzle_id]

    def get_all_puzzle_ids(self, dataset: str = "training") -> List[str]:
        """
        Tüm puzzle ID'lerini al

        Args:
            dataset: "training" veya "evaluation"

        Returns:
            Puzzle ID'leri listesi
        """
        if dataset == "training":
            challenges, _ = self.load_training_data()
        else:
            challenges, _ = self.load_evaluation_data()

        return list(challenges.keys())
