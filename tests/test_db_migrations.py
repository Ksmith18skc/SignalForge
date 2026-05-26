from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

import app.db as db_module
from app.models import MlbEdge


VALIDATION_COLUMNS = {
    "normalized_market_name",
    "market_scope",
    "is_valid",
    "validation_reason",
}


def test_mlb_edge_validation_columns_are_backfilled_on_legacy_table():
    engine = create_engine("sqlite:///:memory:", future=True)
    original_engine = db_module.engine
    db_module.engine = engine
    try:
        _create_legacy_mlb_edges_table(engine)

        db_module._ensure_mlb_edge_validation_columns()

        columns = {col["name"] for col in inspect(engine).get_columns("mlb_edges")}
        assert VALIDATION_COLUMNS.issubset(columns)
    finally:
        db_module.engine = original_engine
        engine.dispose()


def test_mlb_edge_validation_column_migration_is_idempotent():
    engine = create_engine("sqlite:///:memory:", future=True)
    original_engine = db_module.engine
    db_module.engine = engine
    try:
        _create_legacy_mlb_edges_table(engine)

        db_module._ensure_mlb_edge_validation_columns()
        db_module._ensure_mlb_edge_validation_columns()

        columns = [col["name"] for col in inspect(engine).get_columns("mlb_edges")]
        for column in VALIDATION_COLUMNS:
            assert columns.count(column) == 1
    finally:
        db_module.engine = original_engine
        engine.dispose()


def test_new_edge_insert_works_after_validation_column_migration():
    engine = create_engine("sqlite:///:memory:", future=True)
    original_engine = db_module.engine
    db_module.engine = engine
    try:
        _create_legacy_mlb_edges_table(engine)
        db_module._ensure_mlb_edge_validation_columns()

        Session = sessionmaker(bind=engine, autoflush=False, future=True)
        session = Session()
        try:
            edge = MlbEdge(
                game_pk=123,
                edge_type="game_total",
                market="Full Game Total - Under 8.5",
                normalized_market_name="Full Game Total - Under 8.5",
                market_scope="FULL_GAME_TOTAL",
                is_valid=True,
                validation_reason="",
                side="under",
                line=8.5,
                best_book="DraftKings",
                best_price=1.91,
                score=74.0,
                confidence="medium",
                action="Watch",
                chase_risk="low",
                generated_for_date="2026-05-26",
            )
            session.add(edge)
            session.commit()

            stored = session.get(MlbEdge, edge.id)
            assert stored is not None
            assert stored.normalized_market_name == "Full Game Total - Under 8.5"
            assert stored.market_scope == "FULL_GAME_TOTAL"
            assert stored.is_valid is True
        finally:
            session.close()
    finally:
        db_module.engine = original_engine
        engine.dispose()


def _create_legacy_mlb_edges_table(engine) -> None:
    columns = []
    for column in MlbEdge.__table__.columns:
        if column.name in VALIDATION_COLUMNS:
            continue
        ddl = f"{column.name} {column.type.compile(dialect=engine.dialect)}"
        if column.primary_key:
            ddl += " PRIMARY KEY"
        columns.append(ddl)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE mlb_edges ({', '.join(columns)})"))
