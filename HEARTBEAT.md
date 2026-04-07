# Trip Insights on Message

When you message me, I'll analyze `trips.json` and provide insights if relevant:

**Send analysis when:**
- You just added/modified a trip
- Trip approaching (within 7 days)
- Schengen days for someone crosses 80%+ of 90-day limit
- Conflict detected (overlapping trips, person in two places)
- Before summer (June 15+): Calculate impact of current trips on remaining Schengen days through end of summer (Aug 31)

**Stay silent when:**
- No changes since last message
- All trips are far away (>14 days)
- Everything looks normal

**Summer impact analysis:**
- Track Schengen days used per person through Jun 14
- Calculate how many days available for June-Aug (180-day rolling window)
- Alert if usage pattern puts anyone close to 90-day limit during summer
