"""Seed user signals and document type policies.

Revision ID: 005
Revises: 004
Create Date: 2026-01-XX

This migration seeds built-in signals, user-defined signals, document types with fields, and document type policies.
"""
from alembic import op
import sqlalchemy as sa
import json


revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Seed built-in signals, user-defined signals, document types, fields and policies."""
    conn = op.get_bind()
    
    # Seed built-in signals (idempotent - only if they don't exist)
    builtin_signals = [
        ('iban_present', 'IBAN aanwezig', 'Document bevat minimaal één IBAN nummer', 'boolean', 'builtin', 'builtin', None),
        ('date_count', 'Aantal datums', 'Aantal datums (DD-MM-YYYY formaat) in document', 'count', 'builtin', 'builtin', None),
        ('amount_count', 'Aantal bedragen', 'Aantal geldbedragen (€X.XXX,XX formaat) in document', 'count', 'builtin', 'builtin', None),
        ('date_amount_row_count', 'Transactieregels', 'Aantal regels met zowel datum als bedrag', 'count', 'builtin', 'builtin', None),
        ('line_count', 'Aantal regels', 'Aantal niet-lege regels in document', 'count', 'builtin', 'builtin', None),
        ('token_count', 'Aantal woorden', 'Aantal woorden (whitespace-gescheiden tokens)', 'count', 'builtin', 'builtin', None),
    ]
    
    for key, label, description, signal_type, source, compute_kind, config_json in builtin_signals:
        # Check if signal already exists
        result = conn.execute(
            sa.text("SELECT id FROM classification_signals WHERE key = :key"),
            {'key': key}
        )
        if not result.fetchone():
            # Insert built-in signal
            # Escape single quotes in strings
            label_escaped = label.replace("'", "''")
            desc_escaped = (description or '').replace("'", "''")
            config_str = "NULL"
            if config_json:
                config_str = f"'{json.dumps(config_json).replace(chr(39), chr(39)+chr(39))}'"
            
            op.execute(f"""
                INSERT INTO classification_signals (key, label, description, signal_type, source, compute_kind, config_json, created_at, updated_at)
                VALUES ('{key}', '{label_escaped}', '{desc_escaped}', '{signal_type}', '{source}', '{compute_kind}', {config_str}, datetime('now'), datetime('now'))
            """)
    
    # Check if user signals already exist (idempotent)
    result = conn.execute(sa.text("SELECT COUNT(*) FROM classification_signals WHERE source = 'user'"))
    existing_count = result.scalar()
    
    if existing_count == 0:
        # Insert user-defined signals
        # No user-defined signals to seed
        pass
    
    # Seed document types (idempotent - only if they don't exist)
    document_types = [
        {
            'slug': 'aandeelhoudersregister',
            'name': 'Aandeelhoudersregister',
            'description': 'Aandeelhoudersregister is een document dat de lijst van aandeelhouders van een vennootschap bevat.',
            'classification_hints': 'kw:aandelen\nkw:vennootschap\nkw:register\nkw:akte\nkw:gegevens\nkw:per\nkw:kapitaal\nkw:aandeelhouder\nkw:notaris\nkw:levering',
            'extraction_prompt_preamble': 'Extraheer de volgende velden uit het aandeelhoudersregister: aandeelnummer, aandeelhoudernaam, aandeelpercentage, datum van aankoop, aandeelprijs, notarisnaam, leveringsdatum. Verzamel deze gegevens per aandeelhouder en organiseer ze in een lijst.',
            'fields': [],
        },
        {
            'slug': 'bankafschrift',
            'name': 'Bankafschrift',
            'description': 'Bankafschrift (Bij- en afschrijvingen) voor ondernemersrekening, met periode/saldi en transactie-overzicht.',
            'classification_hints': 'kw: rekeningnummer\nkw: saldo\nre: NL\\d{2}[A-Z]{4}\\d{10}\nnot: Commitment Agreement\nnot: Participant',
            'extraction_prompt_preamble': 'Dit document is een Nederlands bankafschrift "Bij- en afschrijvingen" voor een (ondernemers)rekening.\nHerken en extraheer: rekeninghouder, rekeningnummer/ondernemersrekening, periode start/eind, saldo begin/eind, totaal afgeschreven/bijgeschreven, aantallen af/bij, en transacties (datum, omschrijving, bedrag af, bedrag bij).\nTransacties bevatten soms SEPA-details: tegenrekening IBAN, BIC, naam en omschrijving/kenmerk.\nDatums staan in dd-mm-jjjj → retourneer YYYY-MM-DD.\nBedragen staan in Europese notatie (. duizendtallen, , decimalen) en valuta is meestal EUR → retourneer bedragen als decimalen met punt (bijv. 18611.93) en currency EUR.\nGebruik de kolommen "Datum / Omschrijving / Bedrag af / Bedrag bij" als leidraad.',
            'fields': [
                {'key': 'iban', 'label': 'IBAN', 'field_type': 'iban', 'required': True, 'description': None, 'enum_values': None, 'regex': '^[A-Z]{2}\\d{2}(?:\\s*[A-Z0-9]){11,30}$'},
                {'key': 'adres', 'label': 'Adres', 'field_type': 'string', 'required': True, 'description': None, 'enum_values': None, 'regex': None},
                {'key': 'naam', 'label': 'Naam', 'field_type': 'string', 'required': True, 'description': None, 'enum_values': None, 'regex': None},
            ],
        },
        {
            'slug': 'commitment-agreement',
            'name': 'Commitment Agreement',
            'description': 'Een Commitment Agreement is een overeenkomst tussen de Participant (investeerder), de Beheerder en de (be)waarder/stichting waarin de participant zich verplicht om tot een maximaal bedrag ("Commitment") te investeren in een fonds onder de fondsvoorwaarden ("Voorwaarden"). De beheerder kan (delen van) dit commitment opvragen via een betalingsverzoek met een afgesproken betalingstermijn. In de overeenkomst is doorgaans ook vastgelegd welke KYC/AML-documenten de participant moet aanleveren en dat bijlagen onderdeel uitmaken van de overeenkomst.',
            'classification_hints': 'kw: Commitment Agreement\nkw: Participant\nnot: bankafschrift\nnot: rekeningnummer\nnot: bij- en afschrijvingen',
            'extraction_prompt_preamble': 'Je bent een nauwkeurige informatie-extractor voor juridische documenten. \nDoel: haal gestructureerde metadata uit een "Commitment Agreement" en geef uitsluitend JSON terug volgens het opgegeven schema.\n\nInstructies\n- Gebruik alleen informatie die letterlijk in het document staat.\n- Als een veld niet te vinden is: zet de waarde op null (geen aannames).\n- Normaliseer:\n  - Datums naar ISO: YYYY-MM-DD (als dag/maand/jaar ontbreekt -> null).\n  - Bedragen naar number (decimal punt), zonder valuta-tekens; valuta naar aparte "currency" (bijv. "EUR").\n  - IBAN: verwijder spaties en maak uppercase.\n- Houd namen exact zoals geschreven (incl. juridische suffixen zoals B.V., N.V., Stichting).\n- "participant.represents": alleen invullen als er expliciet "namens" / "represented by" / "acting for" staat.\n- KYC/AML velden zijn booleans:\n  - true als het document de eis of aanwezigheid expliciet noemt,\n  - false alleen als het document expliciet zegt dat iets niet vereist/niet van toepassing is,\n  - anders null.\n- "payment_call.notice_period_workdays": vul alleen een getal in als er een expliciet aantal werkdagen staat.\n\nOutput\n- Geef uitsluitend één JSON-object terug dat exact voldoet aan het schema.\n- Geen extra tekst, geen markdown, geen toelichting.',
            'fields': [
                {'key': 'participant', 'label': 'participant', 'field_type': 'string', 'required': True, 'description': None, 'enum_values': None, 'regex': None},
                {'key': 'manager', 'label': 'manager', 'field_type': 'string', 'required': True, 'description': None, 'enum_values': None, 'regex': None},
                {'key': 'bedrag', 'label': 'bedrag', 'field_type': 'money', 'required': True, 'description': None, 'enum_values': None, 'regex': None},
                {'key': 'datum_overeenkomst', 'label': 'datum overeenkomst', 'field_type': 'date', 'required': True, 'description': None, 'enum_values': None, 'regex': None},
                {'key': 'adres', 'label': 'adres', 'field_type': 'string', 'required': True, 'description': None, 'enum_values': None, 'regex': None},
            ],
        },
    ]
    
    for doc_type in document_types:
        # Check if document type already exists
        result = conn.execute(
            sa.text("SELECT id FROM document_types WHERE slug = :slug"),
            {'slug': doc_type['slug']}
        )
        existing = result.fetchone()
        
        if not existing:
            # Insert document type
            # Escape single quotes
            name_escaped = doc_type['name'].replace("'", "''")
            desc_escaped = (doc_type['description'] or '').replace("'", "''")
            hints_escaped = (doc_type['classification_hints'] or '').replace("'", "''")
            preamble_escaped = (doc_type['extraction_prompt_preamble'] or '').replace("'", "''")
            
            op.execute(f"""
                INSERT INTO document_types (name, slug, description, classification_hints, extraction_prompt_preamble, created_at, updated_at)
                VALUES ('{name_escaped}', '{doc_type['slug']}', '{desc_escaped}', '{hints_escaped}', '{preamble_escaped}', datetime('now'), datetime('now'))
            """)
            # Get the inserted ID
            result = conn.execute(
                sa.text("SELECT id FROM document_types WHERE slug = :slug"),
                {'slug': doc_type['slug']}
            )
            doc_type_id = result.scalar()
        else:
            doc_type_id = existing[0]
        
        # Insert fields for this document type (idempotent)
        for field in doc_type['fields']:
            # Check if field already exists
            result = conn.execute(
                sa.text("SELECT id FROM document_type_fields WHERE document_type_id = :doc_type_id AND key = :key"),
                {'doc_type_id': doc_type_id, 'key': field['key']}
            )
            if not result.fetchone():
                # Insert field
                # Escape single quotes
                label_escaped = field['label'].replace("'", "''")
                desc_escaped = (field['description'] or '').replace("'", "''")
                regex_escaped = (field['regex'] or '').replace("'", "''") if field['regex'] else 'NULL'
                enum_vals = "NULL"
                if field['enum_values']:
                    enum_vals = f"'{json.dumps(field['enum_values']).replace(chr(39), chr(39)+chr(39))}'"
                
                op.execute(f"""
                    INSERT INTO document_type_fields 
                    (document_type_id, key, label, field_type, required, description, enum_values, regex, created_at, updated_at)
                    VALUES ({doc_type_id}, '{field['key']}', '{label_escaped}', '{field['field_type']}', {1 if field['required'] else 0}, '{desc_escaped}', {enum_vals}, {f"'{regex_escaped}'" if field['regex'] else 'NULL'}, datetime('now'), datetime('now'))
                """)
    
    # Update policies (idempotent - always updates)
    # Policy for bankafschrift
    policy_json_bankafschrift = json.loads('''{
  "requirements": [
    {
      "signal": "date_count",
      "op": ">=",
      "value": 3
    }
  ],
  "exclusions": [
    {
      "signal": "commitment_agreement",
      "op": "==",
      "value": true
    }
  ],
  "acceptance": {
    "trained_model": {
      "enabled": true,
      "min_confidence": 0.85,
      "min_margin": 0.1
    },
    "deterministic": {
      "enabled": true
    },
    "llm": {
      "enabled": true,
      "require_evidence": true
    }
  }
}''')
    # Update policy - escape JSON string for SQL
    policy_str = json.dumps(policy_json_bankafschrift).replace("'", "''")
    op.execute(f"""
        UPDATE document_types SET classification_policy_json = '{policy_str}' WHERE slug = 'bankafschrift'
    """)


def downgrade() -> None:
    """Remove user-defined signals, document types and clear policies."""
    # Remove user-defined signals (keep built-in signals)
    op.execute("DELETE FROM classification_signals WHERE source = 'user'")
    
    # Remove document types (cascade will remove fields)
    op.execute("DELETE FROM document_types WHERE slug IN ('aandeelhoudersregister', 'bankafschrift', 'commitment-agreement')")
    
    # Clear all policies
    op.execute("UPDATE document_types SET classification_policy_json = NULL")
