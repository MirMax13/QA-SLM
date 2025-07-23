import json
import numpy as np

def calculate_average_bertscore(json_file_path):
    """
    Зчитує JSON файл та обраховує середнє значення BERTScore
    
    Args:
        json_file_path (str): Шлях до JSON файлу
    
    Returns:
        float: Середнє значення BERTScore
    """
    try:
        # Зчитуємо JSON файл
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Витягуємо всі BERTScore значення
        bert_scores = []
        
        for item in data:
            if 'BERTScore' in item:
                score = float(item['BERTScore'].strip('"'))
                bert_scores.append(score)
        
        # Обраховуємо середнє значення
        if bert_scores:
            average_score = np.mean(bert_scores)
            return average_score
        else:
            print("Не знайдено жодного BERTScore значення у файлі")
            return None
            
    except FileNotFoundError:
        print(f"Файл {json_file_path} не знайдено")
        return None
    except json.JSONDecodeError:
        print(f"Помилка при парсингу JSON файлу {json_file_path}")
        return None
    except Exception as e:
        print(f"Виникла помилка: {e}")
        return None

    """
    Детальний аналіз BERTScore значень
    
    Args:
        json_file_path (str): Шлях до JSON файлу
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        bert_scores = []
        
        for item in data:
            if 'BERTScore' in item:
                score_str = item['BERTScore'].strip('"')
                score = float(score_str)
                bert_scores.append(score)
        
        if bert_scores:
            print(f"Загальна кількість прикладів: {len(bert_scores)}")
            print(f"Середнє значення BERTScore: {np.mean(bert_scores):.4f}")
            print(f"Медіана BERTScore: {np.median(bert_scores):.4f}")
            print(f"Мінімальне значення: {np.min(bert_scores):.4f}")
            print(f"Максимальне значення: {np.max(bert_scores):.4f}")
            print(f"Стандартне відхилення: {np.std(bert_scores):.4f}")
            
            # Розподіл за діапазонами
            excellent = sum(1 for score in bert_scores if score >= 0.95)
            good = sum(1 for score in bert_scores if 0.90 <= score < 0.95)
            fair = sum(1 for score in bert_scores if 0.85 <= score < 0.90)
            poor = sum(1 for score in bert_scores if score < 0.85)
            
            print(f"\nРозподіл якості відповідей:")
            print(f"Відмінно (≥0.95): {excellent} ({excellent/len(bert_scores)*100:.1f}%)")
            print(f"Добре (0.90-0.94): {good} ({good/len(bert_scores)*100:.1f}%)")
            print(f"Задовільно (0.85-0.89): {fair} ({fair/len(bert_scores)*100:.1f}%)")
            print(f"Потребує покращення (<0.85): {poor} ({poor/len(bert_scores)*100:.1f}%)")
            
        else:
            print("Не знайдено жодного BERTScore значення у файлі")
            
    except Exception as e:
        print(f"Виникла помилка при аналізі: {e}")

# Основна функція
if __name__ == "__main__":
    file_path = "chatgpt_1.3_new_gpt_val_pred.json"
    
    # Простий розрахунок середнього
    average = calculate_average_bertscore(file_path)
    if average is not None:
        print(f"Середнє значення BERTScore: {average:.4f}")

