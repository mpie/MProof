"""Seed Dutch mortgage document types with fields and validity validation rules.

Revision ID: 007
Revises: 006
Create Date: 2026-05-27

Adds missing mortgage document types (loonstrook, hypotheekakte, arbeidscontract,
identiteitsbewijs) with classification hints, date fields, and date_max_age_days /
date_not_expired validation rules per Dutch NHG/mortgage practice.
"""
from alembic import op
import sqlalchemy as sa
import json


revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None

# Dutch mortgage validity periods (regulatory / market practice)
# taxatierapport: NHG rule – max 6 months (180 days)
# bankafschrift: standard practice – max 3 months (90 days)
# loonstrook: standard practice – max 3 months (90 days)
# arbeidscontract: must not be expired (date_not_expired on datum_einde)
# identiteitsbewijs: valid for 10 years (3650 days), must not be expired


def upgrade() -> None:
    conn = op.get_bind()
    now = "NOW()"

    # ------------------------------------------------------------------ helpers
    def doc_type_id(slug):
        r = conn.execute(sa.text("SELECT id FROM document_types WHERE slug = :s"), {"s": slug}).fetchone()
        return r[0] if r else None

    def ensure_doc_type(slug, name, description, hints, preamble):
        existing = doc_type_id(slug)
        if existing:
            return existing
        conn.execute(sa.text(f"""
            INSERT INTO document_types
              (slug, name, description, classification_hints, extraction_prompt_preamble, created_at, updated_at)
            VALUES (:slug, :name, :desc, :hints, :pre, {now}, {now})
        """), {"slug": slug, "name": name, "desc": description, "hints": hints, "pre": preamble})
        return doc_type_id(slug)

    def field_exists(doc_type_id, key):
        r = conn.execute(
            sa.text("SELECT id FROM document_type_fields WHERE document_type_id = :d AND `key` = :k"),
            {"d": doc_type_id, "k": key}
        ).fetchone()
        return r is not None

    def ensure_field(doc_type_id, key, label, field_type, required=False, description=None, validation_rules=None):
        if field_exists(doc_type_id, key):
            # Update validation_rules if field exists but has none
            if validation_rules is not None:
                conn.execute(sa.text(f"""
                    UPDATE document_type_fields
                    SET validation_rules = :vr, updated_at = {now}
                    WHERE document_type_id = :d AND `key` = :k AND validation_rules IS NULL
                """), {"d": doc_type_id, "k": key, "vr": json.dumps(validation_rules)})
            return
        conn.execute(sa.text(f"""
            INSERT INTO document_type_fields
              (document_type_id, `key`, label, field_type, required, description, validation_rules, created_at, updated_at)
            VALUES (:d, :key, :label, :ft, :req, :desc, :vr, {now}, {now})
        """), {
            "d": doc_type_id, "key": key, "label": label, "ft": field_type,
            "req": required, "desc": description,
            "vr": json.dumps(validation_rules) if validation_rules else None,
        })

    # ------------------------------------------------------------------ taxatierapport
    t_id = doc_type_id("taxatierapport")
    if t_id:
        ensure_field(t_id, "datum_taxatie", "Datum taxatie", "date", required=True,
                     description="Datum waarop de taxatie is uitgevoerd (max 6 maanden geldig voor hypotheekaanvraag)",
                     validation_rules=[{"validation_type": "date_max_age_days", "params": {"max_days": 180}}])
        ensure_field(t_id, "marktwaarde", "Marktwaarde", "money", required=True,
                     description="Marktwaarde van het object in euro's",
                     validation_rules=[{"validation_type": "amount_range", "params": {"min": 50000, "max": 10000000}}])
        ensure_field(t_id, "executiewaarde", "Executiewaarde", "money", required=False,
                     description="Executiewaarde (typisch 85-95% van marktwaarde)",
                     validation_rules=[{"validation_type": "cross_field_ratio",
                                        "params": {"other_field": "marktwaarde", "min_ratio": 0.75, "max_ratio": 0.98}}])
        ensure_field(t_id, "taxateur", "Taxateur / bureau", "string")
        ensure_field(t_id, "adres_object", "Adres getaxeerd object", "string", required=True)

    # ------------------------------------------------------------------ bankafschrift
    b_id = doc_type_id("bankafschrift")
    if b_id:
        ensure_field(b_id, "periode_eind", "Einddatum periode", "date", required=True,
                     description="Einddatum van het bankafschrift (max 3 maanden oud)",
                     validation_rules=[{"validation_type": "date_max_age_days", "params": {"max_days": 90}}])
        ensure_field(b_id, "saldo", "Saldo", "money",
                     description="Saldo aan het einde van de periode")
        ensure_field(b_id, "rekeninghouder", "Rekeninghouder", "string", required=True)

    # ------------------------------------------------------------------ loonstrook
    l_id = ensure_doc_type(
        "loonstrook", "Loonstrook",
        "Maandelijkse salarisspecificatie van de werkgever",
        "kw:loonstrook\nkw:salaris\nkw:brutoloon\nkw:nettoloon\nkw:werkgever\nkw:werknemer\nkw:loonheffing\nkw:sociale verzekering",
        "Dit is een loonstrook. Extraheer: naam werknemer, naam werkgever, bruto maandloon, netto maandloon, datum loonstrook, IBAN van betaling.",
    )
    ensure_field(l_id, "datum", "Datum loonstrook", "date", required=True,
                 description="Datum / periode van de loonstrook (max 3 maanden oud)",
                 validation_rules=[{"validation_type": "date_max_age_days", "params": {"max_days": 90}}])
    ensure_field(l_id, "bruto_maandloon", "Bruto maandloon", "money", required=True,
                 description="Bruto maandloon in euro's",
                 validation_rules=[{"validation_type": "amount_range", "params": {"min": 500, "max": 50000}}])
    ensure_field(l_id, "netto_maandloon", "Netto maandloon", "money")
    ensure_field(l_id, "werkgever", "Werkgever", "string", required=True)
    ensure_field(l_id, "werknemer", "Werknemer", "string", required=True)
    ensure_field(l_id, "iban_uitbetaling", "IBAN uitbetaling", "iban")

    # ------------------------------------------------------------------ hypotheekakte
    h_id = ensure_doc_type(
        "hypotheekakte", "Hypotheekakte",
        "Notariële akte voor het vestigen van een hypotheek",
        "kw:hypotheekakte\nkw:hypothecaire lening\nkw:geldgever\nkw:geldnemer\nkw:onderpand\nkw:notaris\nkw:inschrijving kadaster",
        "Dit is een hypotheekakte. Extraheer: datum akte, naam geldgever, naam geldnemer, hypotheekbedrag, rentepercentage, looptijd, adres onderpand, notaris.",
    )
    ensure_field(h_id, "datum_akte", "Datum akte", "date", required=True,
                 description="Datum van de notariële akte")
    ensure_field(h_id, "hypotheekbedrag", "Hypotheekbedrag", "money", required=True,
                 description="Hoogte van de hypotheek in euro's",
                 validation_rules=[{"validation_type": "amount_range", "params": {"min": 10000, "max": 5000000}}])
    ensure_field(h_id, "rentepercentage", "Rentepercentage", "number",
                 description="Rentepercentage (bijv. 3.5)")
    ensure_field(h_id, "looptijd_jaren", "Looptijd (jaren)", "number")
    ensure_field(h_id, "geldgever", "Geldgever / bank", "string", required=True)
    ensure_field(h_id, "notaris", "Notaris", "string")
    ensure_field(h_id, "adres_onderpand", "Adres onderpand", "string", required=True)

    # ------------------------------------------------------------------ arbeidscontract
    a_id = ensure_doc_type(
        "arbeidscontract", "Arbeidscontract",
        "Arbeidsovereenkomst tussen werkgever en werknemer",
        "kw:arbeidsovereenkomst\nkw:arbeidscontract\nkw:dienstverband\nkw:werkgever\nkw:werknemer\nkw:functie\nkw:salaris\nkw:bepaalde tijd\nkw:onbepaalde tijd",
        "Dit is een arbeidscontract. Extraheer: naam werkgever, naam werknemer, functietitel, startdatum dienstverband, einddatum (indien bepaalde tijd), bruto maandsalaris, type contract (bepaalde/onbepaalde tijd).",
    )
    ensure_field(a_id, "datum_ingang", "Datum ingang", "date", required=True,
                 description="Startdatum van het dienstverband")
    ensure_field(a_id, "datum_einde", "Datum einde", "date", required=False,
                 description="Einddatum bij tijdelijk contract (moet in de toekomst liggen)",
                 validation_rules=[{"validation_type": "date_not_expired", "params": {}}])
    ensure_field(a_id, "bruto_maandsalaris", "Bruto maandsalaris", "money", required=False,
                 description="Bruto maandsalaris in euro's",
                 validation_rules=[{"validation_type": "amount_range", "params": {"min": 500, "max": 50000}}])
    ensure_field(a_id, "type_contract", "Type contract", "string",
                 description="Bepaalde tijd / onbepaalde tijd / oproep")
    ensure_field(a_id, "werkgever", "Werkgever", "string", required=True)
    ensure_field(a_id, "werknemer", "Werknemer", "string", required=True)
    ensure_field(a_id, "functie", "Functie", "string")

    # ------------------------------------------------------------------ identiteitsbewijs
    i_id = ensure_doc_type(
        "identiteitsbewijs", "Identiteitsbewijs / Paspoort",
        "Geldig identiteitsdocument: paspoort, rijbewijs of ID-kaart",
        "kw:paspoort\nkw:identiteitskaart\nkw:rijbewijs\nkw:geldig tot\nkw:BSN\nkw:burgerservicenummer\nkw:nationaliteit",
        "Dit is een identiteitsdocument. Extraheer: voornaam, achternaam, geboortedatum, BSN, documentnummer, geldig tot datum, nationaliteit, type document (paspoort/ID-kaart/rijbewijs).",
    )
    ensure_field(i_id, "geldig_tot", "Geldig tot", "date", required=True,
                 description="Vervaldatum van het identiteitsdocument (moet in de toekomst liggen)",
                 validation_rules=[{"validation_type": "date_not_expired", "params": {}}])
    ensure_field(i_id, "geboortedatum", "Geboortedatum", "date", required=True)
    ensure_field(i_id, "bsn", "BSN", "string", required=False,
                 description="Burgerservicenummer (11-proef)")
    ensure_field(i_id, "documentnummer", "Documentnummer", "string")
    ensure_field(i_id, "naam", "Naam", "string", required=True)
    ensure_field(i_id, "nationaliteit", "Nationaliteit", "string")


def downgrade() -> None:
    conn = op.get_bind()
    # Remove added document types (only the ones we may have created)
    for slug in ("loonstrook", "hypotheekakte", "arbeidscontract", "identiteitsbewijs"):
        r = conn.execute(sa.text("SELECT id FROM document_types WHERE slug = :s"), {"s": slug}).fetchone()
        if r:
            conn.execute(sa.text("DELETE FROM document_type_fields WHERE document_type_id = :d"), {"d": r[0]})
            conn.execute(sa.text("DELETE FROM document_types WHERE id = :d"), {"d": r[0]})
    # For existing types (taxatierapport, bankafschrift), we only added fields — leave them
