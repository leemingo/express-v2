"""Schema for SPADL actions.

This module defines the data schema for LSDP (Labelled Soccer Data Protocol) 
dataframes using Pandera for type validation and constraint checking.
"""

from typing import Any, Optional

import pandera as pa
from pandera.typing import Series
import assertion.config as lsdpconfig

class LSDPSchema(pa.SchemaModel):
    """Definition of a LSDP dataframe."""

    game_id: Series[Any] = pa.Field()
    original_event_id: Series[Any] = pa.Field(nullable=True)
    action_id: Series[int] = pa.Field()
    period_id: Series[int] = pa.Field(ge=1, le=5)
    time_seconds: Series[float] = pa.Field(ge=0) # Seconds since the start of the period.
    relative_time_seconds: Series[float] = pa.Field(ge=0) # Arrive Seconds since the pass event.  

    team_id: Series[int] = pa.Field()
    player_id: Series[int] = pa.Field()
    relative_player_id: Series[int] = pa.Field()
    reactor_team_id: Series[int] = pa.Field()
    reactor_player_id: Series[int] = pa.Field()

    start_x: Series[float] = pa.Field(ge=0, le=lsdpconfig.field_length)
    start_y: Series[float] = pa.Field(ge=0, le=lsdpconfig.field_width)
    end_x: Series[float] = pa.Field(ge=0, le=lsdpconfig.field_length)
    end_y: Series[float] = pa.Field(ge=0, le=lsdpconfig.field_width)
    dx: Series[float] = pa.Field()
    dy: Series[float] = pa.Field()
    
    type_id: Series[int] = pa.Field(isin=lsdpconfig.actiontypes_df().type_id)
    type_name: Optional[Series[str]] = pa.Field(isin=lsdpconfig.actiontypes_df().type_name)
    result_name: Optional[Series[str]] = pa.Field()
    relative_id: Series[int] = pa.Field()
    pair_id: Series[int] = pa.Field()

    class Config:  # noqa: D106
        strict = True
        coerce = True
