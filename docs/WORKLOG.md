# Выполненная работа — 2026-09-08

## Объём

Этапы 1 и 2 реализованы для выбранного продукта v0.1: один инструмент,
long-only confluence, исторический бэктест и paper replay. Исходный репозиторий
не модифицировался. Это новая реализация необходимых компонентов, без копирования
всех экспериментальных скриптов и их библиотек. Решение о составе подробно
зафиксировано в MIGRATION.md; невыполненные production-этапы — в ROADMAP.md.

## Этап 1

- [x] Отдельный Python package RoboMOEX с CLI и версией 0.1.0.
- [x] Python 3.12.14, закреплённые прямые и транзитивные зависимости, uv.lock.
- [x] requirements.lock с хешами для установки runtime через pip/uv pip.
- [x] Явная конфигурация стратегии/исполнения; ошибки имён и значений отвергаются.
- [x] Режимы backtest/paper/live. По умолчанию offline paper demo.
- [x] Live заблокирован программно до обращений к файлам или сети.
- [x] Удалена необходимость торговых токенов и heavyweight ML/GPU библиотек.
- [x] README, миграция, контракты, остаточный план и MIT attribution.
- [x] CI workflow для Windows/Linux с закреплёнными action SHA.

## Этап 2

- [x] Контракт закрытой минутной свечи и явного session schedule.
- [x] Валидация timezone/OHLC/дублей/пропусков/перекрытий/актуальности.
- [x] Причинное совмещение 1m/15m/дневной выбранной сессии по close_time.
- [x] Прогрев без bfill; RSI edge cases и ATR gap calculation.
- [x] Один симулятор для paper/backtest; вход на следующем open.
- [x] SL/TP независимо от фильтров, adverse intrabar policy, gap fills.
- [x] Лотность, шаг цены, комиссия, slippage, корректный cash/equity/PnL.
- [x] Корректный знак drawdown и явно определённая дневная annualization.
- [x] Будущий inference на полном контексте с проверкой forecast start.
- [x] Sample-path denormalization до расчёта квантилей цен.
- [x] MOEX ISS pagination, bounded retries, отсутствие скрытого пропуска ошибок.
- [x] Атомарный проверяемый cache; идентичные расчёты до/после чтения CSV.
- [x] Тесты эталонных сделок и invariance при обрезке/изменении будущей истории.

## Проверки

Локальная платформа: Windows, Python 3.12.14, pandas 3.0.1, numpy 2.3.5.

| Проверка | Результат |
|---|---|
| pytest | 55 тестов; окончательный результат в verification.json |
| Coverage | Порог CI 85%; фактический результат в verification.json |
| Ruff check / format | Проверены исходники и тесты |
| uv build --no-build-isolation | Собраны sdist и wheel |
| Установка wheel в отдельный venv | Только runtime-зависимости из hash-locked файла; demo выполнен |
| Paper/backtest parity | Совпадают сделки, equity и метрики на одной истории |
| Повторяемость | Повторный demo даёт тот же report; cache round-trip сохраняет hash |
| Реальный read-only MOEX smoke | SBER/TQBR, 2026-08-20 10:00–10:05 MSK: получены 5 минутных свечей |
| Broker API | Не вызывался; заявок не отправлялось |

При проверке cache parity обнаружена потеря последних разрядов float при
стандартном CSV parsing. Чтение цен изменено на float_precision='round_trip';
регрессионный тест сравнивает полный report до и после чтения cache.

CI-конфигурация не равнозначна выполненному удалённому CI. Локально проверена
Windows-сборка; результат GitHub Actions/Linux фиксируется отдельно после
создания и загрузки репозитория. Не проводились многодневный market soak,
обучение/загрузка Lag-Llama checkpoint, тесты доходности или настоящих исполнений.

## Воспроизведение

```sh
uv sync --frozen --extra dev
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen pytest --cov=robomoex --cov-fail-under=85
uv run --frozen robomoex --demo
uv build --no-build-isolation
```

Необязательная установка готового wheel в чистую среду:

```sh
python -m venv .wheel-venv
# После активации этой среды:
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps dist/robomoex-0.1.0-py3-none-any.whl
python -I -m robomoex.cli --demo
```

requirements.lock экспортируется: `uv export --frozen --no-dev --no-emit-project
--format requirements-txt --output-file requirements.lock`.
