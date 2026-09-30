import fastf1
fastf1.set_log_level("WARNING")
fastf1.Cache.enable_cache("data/cache")
s = fastf1.get_session(2025, "Italian", "R")
s.load(laps=True, weather=True, messages=True, telemetry=False)
print(s.laps.shape, sorted(s.laps["Compound"].dropna().unique()))
print(s.laps.columns.tolist())
print(s.race_control_messages.shape)
print(s.race_control_messages.head(5).to_string())
