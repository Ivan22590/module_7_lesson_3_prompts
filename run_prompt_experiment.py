#!/usr/bin/env python3
"""
Скрипт для изучения качества промптов.

Цель: собрать промпт, получить ответ модели, улучшить промпт (few-shot),
сравнить два ответа и собрать артефакт — README.md для пользователя.

Бытовая мини-задача: напоминание пить воду (ежедневный план-график).
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import LLMClient, system_message, user_message

# ============================================================
# Параметры эксперимента
# ============================================================

MODEL = "openai/gpt-4.1-mini"
TEMPERATURE = 0.2

# ============================================================
# Промпты
# ============================================================

SYSTEM_PROMPT_1 = (
    "Ты — персональный помощник по формированию полезных привычек. "
    "Твоя задача — давать чёткие, практические и лаконичные рекомендации. "
    "Отвечай только в указанном JSON-формате."
)

USER_PROMPT_1 = (
    "Мне нужно составить план-график напоминаний пить воду в течение рабочего дня. "
    "Я работаю с 9:00 до 18:00, хочу выпивать минимум 8 стаканов воды. "
    "Создай расписание с конкретными временем и рекомендациями.\n\n"
    "Ответ дай строго в формате JSON:\n"
    "{\n"
    '  "title": "строка — название плана",\n'
    '  "steps": ["строка — шаг 1", "строка — шаг 2", ...],\n'
    '  "notes": ["строка — заметка 1", "строка — заметка 2", ...]\n'
    "}\n"
    "Не добавляй никакого текста вне JSON-блока."
)

SYSTEM_PROMPT_2 = (
    "Ты — UX-исследователь привычек. Ты создаёшь микро-планы действий, "
    "которые люди реально будут выполнять. "
    "Твои шаги — конкретные, измеримые, с привязкой ко времени. "
    "Твои заметки — краткие (до 12 слов каждая), без воды. "
    "Отвечай ТОЛЬКО валидным JSON, без markdown-обёрток."
)

USER_PROMPT_2 = (
    "Мне нужно составить план-график напоминаний пить воду в течение рабочего дня. "
    "Я работаю с 9:00 до 18:00, хочу выпивать минимум 8 стаканов воды. "
    "Создай расписание с конкретным временем и рекомендациями.\n\n"
    "Ответ дай строго в формате JSON:\n"
    "{\n"
    '  "title": "строка — название плана",\n'
    '  "steps": ["строка — шаг 1", "строка — шаг 2", ...],\n'
    '  "notes": ["строка — заметка 1", "строка — заметка 2", ...]\n'
    "}\n\n"
    "Пример хорошего ответа:\n"
    '{\n'
    '  "title": "План гидратации на рабочий день",\n'
    '  "steps": [\n'
    '    "09:00 — стакан воды сразу после прихода на рабочее место",\n'
    '    "10:30 — стакан воды перед началом активной фазы работы",\n'
    '    "12:00 — стакан воды за 10 минут до обеда",\n'
    '    "13:30 — стакан воды сразу после обеда",\n'
    '    "15:00 — стакан воды после дневного спада энергии",\n'
    '    "16:30 — стакан воды перед завершением рабочих задач",\n'
    '    "17:30 — стакан воды в конце рабочего дня"\n'
    '  ],\n'
    '  "notes": [\n'
    '    "Держите бутылку воды на столе как визуальный триггер",\n'
    '    "Если чувствуете усталость — выпейте воду вместо кофе",\n'
    '    "Используйте приложение-трекер для подсчёта стаканов"\n'
    '  ]\n'
    "}"
)

# ============================================================
# Тексты для README
# ============================================================

EXPERIMENT_GOAL = (
    "Собрать управляемый промпт по схеме "
    "роль → контекст → задача → формат ответа, получить стабильный "
    "структурированный ответ модели и превратить его в полезный артефакт. "
    "Отдельно проверить, даёт ли одна итерация качества (уточнение роли + "
    "few-shot пример) измеримо лучший результат при неизменных параметрах "
    "генерации."
)

EXPERIMENT_STEPS = [
    "Выбрана бытовая мини-задача: план напоминаний пить воду "
    "в течение рабочего дня (9:00–18:00, минимум 8 стаканов).",
    "Собран базовый промпт (запуск 1): роль — «персональный помощник по "
    "полезным привычкам», контекст — рабочий день и цель по объёму воды, "
    "задача — составить расписание, формат — строгий JSON {title, steps, notes}.",
    "Запуск выполнен через модуль llm (LLMClient.chat): модель "
    "openai/gpt-4.1-mini, temperature 0.2, системная инструкция отправлена "
    "отдельным сообщением с ролью system.",
    "Модель по умолчанию (openai/gpt-6-luna) отвергла temperature 0.2 с "
    "ошибкой HTTP 400 — выбрана openai/gpt-4.1-mini, которая поддерживает "
    "этот параметр.",
    "Сделана одна итерация качества (запуск 2): роль уточнена до "
    "«UX-исследователя привычек», добавлены ограничения (шаги измеримые и с "
    "привязкой ко времени, заметки до 12 слов, ответ без markdown-обёрток) и "
    "добавлен few-shot пример готового JSON.",
    "Повторный запуск выполнен с теми же моделью и temperature, чтобы "
    "сравнение было корректным — менялся только текст промпта.",
    "Все запросы, ответы и параметры сохранены в "
    "prompt_experiment_results.json; JSON извлекается из ответа с очисткой от "
    "markdown-обёртки.",
    "Ответы сравнены по трём критериям: соответствие JSON-формату, "
    "полезность шагов (steps), лаконичность заметок (notes).",
    "Итоговый артефакт — этот README: инструкция собрана из лучшего ответа.",
]


# ============================================================
# Утилиты
# ============================================================

def extract_json(text: str) -> dict | None:
    """Извлекает JSON-объект из ответа модели (может содержать markdown-обёртку)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@dataclass
class RunRecord:
    """Запись одного запуска промпта."""
    run_number: int
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float
    response_raw: str
    response_json: dict | None
    model_name: str | None
    usage: dict | None


def save_results(records: list[RunRecord], output_path: Path) -> None:
    """Сохраняет все записи в JSON-файл."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "runs": [
            {
                "run_number": r.run_number,
                "model": r.model,
                "temperature": r.temperature,
                "system_prompt": r.system_prompt,
                "user_prompt": r.user_prompt,
                "response_raw": r.response_raw,
                "response_json": r.response_json,
                "model_used": r.model_name,
                "usage": r.usage,
            }
            for r in records
        ],
    }
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Результаты сохранены: {output_path}")


def load_records(path: Path) -> list[RunRecord]:
    """Восстанавливает список записей из сохранённого JSON-файла."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[RunRecord] = []
    for item in data["runs"]:
        records.append(
            RunRecord(
                run_number=item["run_number"],
                system_prompt=item["system_prompt"],
                user_prompt=item["user_prompt"],
                model=item["model"],
                temperature=item["temperature"],
                response_raw=item["response_raw"],
                response_json=item["response_json"],
                model_name=item.get("model_used"),
                usage=item.get("usage"),
            )
        )
    return records


def report(records: list[RunRecord]) -> None:
    """Сравнивает ответы и генерирует README.md."""
    r1, r2 = records[0], records[1]

    # ---------- Сравнение ----------
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ОТВЕТОВ")
    print("=" * 60)

    def assess(record: RunRecord) -> dict:
        valid_json = record.response_json is not None
        steps = record.response_json.get("steps", []) if record.response_json else []
        notes = record.response_json.get("notes", []) if record.response_json else []
        return {
            "valid_json": valid_json,
            "steps_count": len(steps),
            "steps_useful": len(steps) >= 4,
            "notes_count": len(notes),
            "notes_concise": len(notes) <= 3,
        }

    a1, a2 = assess(r1), assess(r2)

    c = {
        "json_format": {
            "r1": "ДА" if a1["valid_json"] else "НЕТ",
            "r2": "ДА" if a2["valid_json"] else "НЕТ",
        },
        "steps_useful": {
            "r1": "ДА" if a1["steps_useful"] else "НЕТ",
            "r2": "ДА" if a2["steps_useful"] else "НЕТ",
        },
        "notes_concise": {
            "r1": "ДА" if a1["notes_concise"] else "НЕТ",
            "r2": "ДА" if a2["notes_concise"] else "НЕТ",
        },
    }

    print(f"JSON формат:")
    print(f"  Запуск 1 (базовый):     {c['json_format']['r1']}")
    print(f"  Запуск 2 (few-shot):    {c['json_format']['r2']}")
    print(f"Полезность шагов (>=4):")
    print(f"  Запуск 1 (базовый):     {c['steps_useful']['r1']}")
    print(f"  Запуск 2 (few-shot):    {c['steps_useful']['r2']}")
    print(f"Лаконичность notes (<=3):")
    print(f"  Запуск 1 (базовый):     {c['notes_concise']['r1']}")
    print(f"  Запуск 2 (few-shot):    {c['notes_concise']['r2']}")

    # ---------- Генерация README.md ----------
    best_json = r2.response_json if (
        r2.response_json and a2["valid_json"] and a2["steps_count"] >= a1["steps_count"]
    ) else r1.response_json

    if best_json:
        readme_lines = [
            f"# {best_json.get('title', 'План гидратации')}",
            "",
            "## Инструкция по шагам",
            "",
        ]
        for i, step in enumerate(best_json.get("steps", []), 1):
            readme_lines.append(f"{i}. {step}")

        readme_lines.extend(["", "## Заметки", ""])
        for note in best_json.get("notes", []):
            readme_lines.append(f"- {note}")

        readme_lines.extend(
            [
                "",
                "---",
                "",
                "## Цель исследования",
                "",
            ]
            + EXPERIMENT_GOAL.splitlines(True)
            + [
                "",
                "---",
                "",
                "## Ход исследования",
                "",
            ]
            + [f"{i}. {s}" for i, s in enumerate(EXPERIMENT_STEPS, 1)]
            + [
                "",
                "---",
                "",
                "## Сравнение ответов (до / после итерации)",
                "",
                "| Критерий              | Базовый промпт    | Few-shot промпт   |",
                "|-----------------------|-------------------|-------------------|",
                f"| JSON-формат           | {c['json_format']['r1']:<17} | {c['json_format']['r2']:<17} |",
                f"| Полезность steps      | {c['steps_useful']['r1']:<17} | {c['steps_useful']['r2']:<17} |",
                f"| Лаконичность notes    | {c['notes_concise']['r1']:<17} | {c['notes_concise']['r2']:<17} |",
                "",
                "---",
                "",
                f"*Сгенерировано {datetime.now().strftime('%d.%m.%Y')} на основе результата LLM-эксперимента. Модель: {MODEL}, temperature: {TEMPERATURE}.*",
            ]
        )

        readme_path = Path("README.md")
        readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
        print(f"\nREADME.md сохранён: {readme_path}")
    else:
        print("\nНе удалось извлечь JSON из обоих ответов — README.md не создан.")


# ============================================================
# Основной поток
# ============================================================

def main() -> None:
    client = LLMClient()
    records: list[RunRecord] = []

    # ---------- Запуск 1: базовый промпт ----------
    print("=" * 60)
    print("ЗАПУСК 1: базовый промпт (без few-shot)")
    print("=" * 60)

    result_1 = client.chat(
        [system_message(SYSTEM_PROMPT_1), user_message(USER_PROMPT_1)],
        model=MODEL,
        temperature=TEMPERATURE,
    )

    json_1 = extract_json(result_1.content)

    record_1 = RunRecord(
        run_number=1,
        system_prompt=SYSTEM_PROMPT_1,
        user_prompt=USER_PROMPT_1,
        model=MODEL,
        temperature=TEMPERATURE,
        response_raw=result_1.content,
        response_json=json_1,
        model_name=result_1.model,
        usage=result_1.usage,
    )
    records.append(record_1)

    print(f"Модель: {result_1.model}")
    print(f"Токены: {result_1.usage}")
    print(f"\nОтвет (сырой):\n{result_1.content}")
    print(
        f"\nОтвет (JSON):\n"
        f"{json.dumps(json_1, ensure_ascii=False, indent=2) if json_1 else 'НЕ УДАЛОСЬ'}"
    )

    # ---------- Запуск 2: улучшенный промпт (few-shot) ----------
    print("\n" + "=" * 60)
    print("ЗАПУСК 2: улучшенный промпт (с few-shot примером)")
    print("=" * 60)

    result_2 = client.chat(
        [system_message(SYSTEM_PROMPT_2), user_message(USER_PROMPT_2)],
        model=MODEL,
        temperature=TEMPERATURE,
    )

    json_2 = extract_json(result_2.content)

    record_2 = RunRecord(
        run_number=2,
        system_prompt=SYSTEM_PROMPT_2,
        user_prompt=USER_PROMPT_2,
        model=MODEL,
        temperature=TEMPERATURE,
        response_raw=result_2.content,
        response_json=json_2,
        model_name=result_2.model,
        usage=result_2.usage,
    )
    records.append(record_2)

    print(f"Модель: {result_2.model}")
    print(f"Токены: {result_2.usage}")
    print(f"\nОтвет (сырой):\n{result_2.content}")
    print(
        f"\nОтвет (JSON):\n"
        f"{json.dumps(json_2, ensure_ascii=False, indent=2) if json_2 else 'НЕ УДАЛОСЬ'}"
    )

    # ---------- Сохранение результатов, сравнение и README ----------
    output_path = Path("prompt_experiment_results.json")
    save_results(records, output_path)
    report(records)

    client.close()
    print("\nГотово!")


if __name__ == "__main__":
    if "--readme-only" in sys.argv:
        # Пересобрать README из сохранённых результатов, без обращений к API
        report(load_records(Path("prompt_experiment_results.json")))
    else:
        main()
