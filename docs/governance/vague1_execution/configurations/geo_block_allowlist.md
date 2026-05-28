# Geo-block — Allow-list FR + BE + CH + LU

Configuration du middleware geo-block conforme à la stratégie bootstrap légal (4 juridictions au lieu de 30+).

---

## Mode

`GEO_BLOCK_MODE=allowlist` — refuse par défaut, autorise explicitement.

## Pays autorisés (ISO 3166-1 alpha-2)

| Code | Pays | Justification |
|---|---|---|
| **FR** | France | Marché principal, droit conso maîtrisé |
| **BE** | Belgique | Francophone, droit similaire FR, UE |
| **CH** | Suisse | Francophone (CH-romande), hors UE, simplification TVA |
| **LU** | Luxembourg | Francophone, UE, marché financier réceptif |

**Volume marché adressable estimé** : ~3 000 - 5 000 traders retail XAU francophones (eval_25 wedge FR + extensions BE/CH/LU).

## Détection IP

Service GeoIP recommandé : **MaxMind GeoLite2** (free, EU-friendly) ou **Cloudflare Workers GeoIP** (si Cloudflare devant).

```python
# backend/src/api/middleware/geo_block.py
from fastapi import Request, HTTPException
from geoip2.database import Reader as GeoIPReader

ALLOWED_COUNTRIES = {"FR", "BE", "CH", "LU"}

geo_reader = GeoIPReader("./data/geolite2/GeoLite2-Country.mmdb")

async def geo_block_middleware(request: Request, call_next):
    if not GEO_BLOCK_ENABLED:
        return await call_next(request)

    client_ip = request.headers.get("CF-Connecting-IP") or request.client.host

    try:
        response = geo_reader.country(client_ip)
        country_code = response.country.iso_code
    except Exception:
        # Si lookup échoue, on accepte (fail-open pour pas bloquer les ipv6 mal lookupées)
        return await call_next(request)

    if country_code not in ALLOWED_COUNTRIES:
        # Sauf si le path est /api/v1/admin/health pour monitoring externe
        if request.url.path.startswith("/api/v1/health"):
            return await call_next(request)
        raise HTTPException(
            status_code=451,
            detail={
                "error": "geographic_restriction",
                "message": "M.I.A. Markets est en phase d'accès anticipé restreinte à FR, BE, CH, LU.",
                "country_detected": country_code,
                "list_url": "/restricted-region",
            },
        )

    return await call_next(request)
```

## Page `/restricted-region`

À créer côté frontend : page publique honnête expliquant la restriction et offrant inscription à waitlist.

```tsx
// frontend/app/restricted-region/page.tsx
export default function RestrictedRegionPage() {
  return (
    <main className="container mx-auto px-7 py-20 text-center max-w-2xl">
      <h1 className="text-3xl font-semibold mb-4">Service en accès anticipé restreint</h1>
      <p className="text-text-secondary mb-6">
        M.I.A. Markets est en phase d'accès anticipé.
        L'inscription est actuellement limitée aux résidents de :
      </p>
      <ul className="text-text-primary font-semibold space-y-1 mb-8">
        <li>🇫🇷 France</li>
        <li>🇧🇪 Belgique</li>
        <li>🇨🇭 Suisse</li>
        <li>🇱🇺 Luxembourg</li>
      </ul>
      <p className="text-text-muted text-sm mb-8">
        Si vous résidez dans l'un de ces pays mais voyez cette page, il peut s'agir
        d'une erreur de géolocalisation ou de l'usage d'un VPN. Contactez-nous :
        <a href="mailto:contact@mia.markets" className="text-accent ml-1">contact@mia.markets</a>
      </p>
      <hr className="border-border my-8" />
      <h2 className="text-xl font-semibold mb-3">Inscrivez-vous à la liste d'attente</h2>
      <p className="text-text-secondary mb-4">
        Nous étendrons progressivement la disponibilité.
        Soyez notifié dès que votre pays est inclus.
      </p>
      <form className="flex flex-col sm:flex-row gap-2 justify-center">
        <input type="email" placeholder="votre@email.com" className="px-4 py-2 bg-bg-elevated border border-border-light rounded" />
        <button className="btn-primary px-6 py-2">Me notifier</button>
      </form>
    </main>
  );
}
```

## Tests

```python
# backend/tests/test_geo_block.py
@pytest.mark.parametrize("country,expected", [
    ("FR", 200),  # allowed
    ("BE", 200),
    ("CH", 200),
    ("LU", 200),
    ("US", 451),  # blocked
    ("UK", 451),
    ("GB", 451),
    ("CA", 451),
    ("JP", 451),
])
async def test_geo_block_allowlist(country, expected):
    with patch_geoip(country=country):
        response = await client.get("/api/v1/lectures/test")
    assert response.status_code == expected
```

## VPN

Test manuel obligatoire **avant DG-043 Stripe live** :
- VPN US → doit voir page `/restricted-region`, pas de CTA paiement
- VPN UK → idem
- VPN FR depuis machine étrangère → doit fonctionner normalement
- Tor exit node US → bloqué (peut être contourné, accepté pour V0)

## Évolution V2+

Quand budget avocat fintech engagé (M3) + revenue stable :
- Élargir UE complète (27 pays UE)
- Configurer Stripe Tax OSS pour TVA intracommunautaire
- Décider US (KYC + SEC alignement très lourd) — probablement non en V2

## Note importante

**Le geo-block est une couche de protection, pas une sécurité absolue.** Un utilisateur déterminé peut utiliser un VPN. La stratégie repose sur :

1. **Couche 1** : geo-block IP (90 % des cas)
2. **Couche 2** : déclaration résidence fiscale au signup (case à cocher obligatoire)
3. **Couche 3** : adresse de facturation Stripe doit être dans pays autorisé
4. **Couche 4** : CGU article X+4 stipule que le contournement (VPN) constitue un manquement et entraîne résiliation sans remboursement

Couche 1 + couche 3 (Stripe) couvrent ≥ 95 % des risques.
