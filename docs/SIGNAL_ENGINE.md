> SUPERSEDED: see REFACTOR_AUDIT.md and results/context_cache_v2/status.json.
> Earlier claims about verified ex-dates, calendar, zero spread and readiness are invalid.

# Новый сигнальный движок SNGS

Добавлен минимальный explainable engine в `robomoex.signal_engine`.

## Что реализовано

- причинные признаки: log returns 1/5/20, EMA20/50/200, RSI14, ATR14,
  z-score, realised volatility, volume/turnover;
- внешние контекстные признаки при наличии: IMOEX, нефть, USD/RUB, CNY/RUB,
  RGBI/ставка, spread, event penalty и event risk;
- winsorization и rolling robust z-score без использования будущих наблюдений;
- композитный score `[-1, 1]` с трендом, momentum, нефтью, валютой, ставкой,
  relative strength и ликвидностью;
- hysteresis для BUY/EXIT и безопасная деградация в NO_TRADE;
- блокировка при высокой важности события или плохой ликвидности;
- риск-лимиты: 0.3% на сделку, 5% позиции, стоп 2.5 ATR, trailing 3.5 ATR,
  time stop 7 баров, спред 25 bps и оборот 30 млн рублей;
- размер позиции выбирает меньший из risk-based и target-volatility методов;
- JSON-контракт сигнала содержит score, confidence, target weight, stop,
  режим рынка, риск события, причины и предупреждения.

## Ограничения текущей версии

Дневные адаптеры IMOEX, RGBI, USD/RUB и BRFOB подключаются через
`robomoex-context` и сохраняются в инкрементальном кешe. Стакан, сделки,
новости, корпоративные действия и доступность займа остаются внешними
источниками. При отсутствии контекста движок не подставляет значения и
снижает сигнал до безопасного решения.

Перед включением в торговый путь нужен корпоративный календарь событий с
временем публикации, учёт дивидендных гэпов и периодический walk-forward
бэктест с чувствительностью к издержкам.
