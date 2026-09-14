import { test, expect, type APIRequestContext } from '@playwright/test';

/**
 * PAY-3 (G1) — LE TEST VIVANT. La défense contre le scénario le plus coûteux.
 *
 * Le scénario : le webhook cesse de mettre à jour l'état, le client paie, Stripe
 * est content, l'app le croit non abonné, personne ne le sait. Ça ne déclenche
 * aucune alarme ; ça se lit comme une baisse de conversion qu'on impute à autre
 * chose pendant des semaines.
 *
 * CE QUI EST MESURÉ ICI
 * ---------------------
 * La chaîne qui nous appartient, de bout en bout, avec du VRAI Stripe :
 *
 *   abonnement créé chez Stripe (API réelle, mode test)
 *     → Stripe émet de VRAIS événements signés
 *       → `stripe listen` les relaie vers /api/billing/webhook
 *         → notre vérification de signature les accepte
 *           → l'accès s'ouvre, et la route de données cesse de refuser
 *
 * Aucun faux client Stripe nulle part. C'est le seul moyen d'attraper la classe
 * de panne qui compte : un `verify_webhook` renvoyant un objet `stripe.Event`
 * (dont `.get()` lève) est passé en production précisément parce que le faux
 * client, lui, renvoyait un `dict`.
 *
 * POURQUOI PAS LA PAGE CHECKOUT
 * -----------------------------
 * Une version antérieure pilotait la page Checkout hébergée avec la carte 4242.
 * Constat après plusieurs passages : Stripe refuse la soumission depuis un
 * navigateur automatisé — la page présente même une case « I am an AI agent
 * acting on behalf of someone else », et la session reste `status: open`,
 * `payment_status: unpaid`. C'est une protection anti-robot délibérée de Stripe,
 * pas une panne du produit, et ce n'est pas une surface qu'on cherche à
 * contourner. La page Checkout appartient à Stripe ; ce qui nous appartient,
 * c'est ce qui se passe APRÈS le paiement — exactement ce qui est testé ici. Le
 * parcours visuel complet reste couvert par le passage manuel du fondateur, qui
 * est de toute façon la condition de fusion.
 *
 * CE QU'IL NE FAIT SURTOUT PAS
 * ----------------------------
 * Il n'appelle JAMAIS `POST /api/billing/sync`. La réconciliation directe
 * (PAY-3e) sauverait le client et rendrait le test vert alors même que le
 * webhook est mort — elle masquerait la panne que ce test existe pour détecter.
 * Seul le chemin webhook est mesuré.
 *
 * PRÉREQUIS (clés de TEST uniquement)
 * -----------------------------------
 *   STRIPE_SECRET_KEY      sk_test_… (une clé live fait échouer le test exprès)
 *   STRIPE_PRICE_MONTHLY   price_… récurrent, dans le MÊME mode
 *   PAY3_API_BASE          base du backend, ex. http://127.0.0.1:8000
 *   et, côté backend lancé pour le test :
 *     STRIPE_WEBHOOK_SECRET issu de `stripe listen`, SUBSCRIPTION_GATE_ENFORCED=1,
 *     EMAIL_VERIFICATION_ENFORCED=0 (pas de SMTP dans un runner).
 */

// Combien de temps on accorde au webhook avant de considérer que le client est
// enfermé dehors. Généreux pour un runner CI, mais fini : au-delà, un vrai
// client aurait abandonné.
const WEBHOOK_DEADLINE_MS = 90_000;
const POLL_INTERVAL_MS = 2_000;

const STRIPE_API = 'https://api.stripe.com';

const REQUIRED = process.env.PAY3_LIVE_REQUIRED === '1';
const SECRET = process.env.STRIPE_SECRET_KEY ?? '';
const PRICE_ID = process.env.STRIPE_PRICE_MONTHLY ?? '';
const API_BASE = (process.env.PAY3_API_BASE ?? '').replace(/\/$/, '');

const missing: string[] = [];
if (!SECRET) missing.push('STRIPE_SECRET_KEY');
if (!PRICE_ID) missing.push('STRIPE_PRICE_MONTHLY');
if (!API_BASE) missing.push('PAY3_API_BASE');

test.beforeAll(() => {
  // Garde-fou avant tout le reste : ce test crée de vrais objets Stripe et
  // déclenche un vrai débit. Sur une clé live, ce serait de l'argent réel.
  if (SECRET.startsWith('sk_live_')) {
    throw new Error(
      'PAY-3 live test refusé : STRIPE_SECRET_KEY est une clé LIVE. ' +
        'Ce test crée un abonnement facturé et ne doit JAMAIS toucher un compte ' +
        'réel. Utilise une clé sk_test_.',
    );
  }
  if (REQUIRED && missing.length > 0) {
    throw new Error(
      `PAY-3 live test EXIGÉ mais non configuré — variables manquantes : ${missing.join(', ')}. ` +
        "Ce test est la seule défense automatisée contre « payé mais pas d'accès » ; " +
        'le laisser se désactiver silencieusement rendrait le déploiement aveugle.',
    );
  }
});

test.skip(
  !REQUIRED && missing.length > 0,
  `PAY-3 live test non configuré (${missing.join(', ')}) — pose PAY3_LIVE_REQUIRED=1 pour en faire un échec.`,
);

function uniqueEmail(): string {
  const stamp = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  return `pay3-e2e+${stamp}@mia-markets-test.invalid`;
}

/** Appel à l'API Stripe RÉELLE (mode test), form-encoded comme Stripe l'attend. */
async function stripe(
  api: APIRequestContext,
  path: string,
  form?: Record<string, string>,
): Promise<Record<string, unknown>> {
  const resp = await api.post(`${STRIPE_API}${path}`, {
    headers: { Authorization: `Bearer ${SECRET}` },
    form: form ?? {},
  });
  const body = (await resp.json()) as Record<string, unknown>;
  if (!resp.ok()) {
    const err = (body.error ?? {}) as { message?: string; code?: string };
    throw new Error(
      `Stripe ${path} a répondu ${resp.status()} : ${err.message ?? JSON.stringify(body)}` +
        (err.code ? ` (code ${err.code})` : ''),
    );
  }
  return body;
}

/**
 * Interroge l'état d'abonnement jusqu'à ce que l'accès s'ouvre — PAR LE WEBHOOK.
 * Aucun appel à /sync : c'est tout l'intérêt.
 */
async function waitForWebhookAccess(
  api: APIRequestContext,
): Promise<{ granted: boolean; elapsedMs: number; last: unknown }> {
  const started = Date.now();
  let last: unknown = null;
  while (Date.now() - started < WEBHOOK_DEADLINE_MS) {
    const resp = await api.get(`${API_BASE}/api/billing/subscription`);
    if (resp.ok()) {
      last = await resp.json();
      if ((last as { has_access?: boolean }).has_access === true) {
        return { granted: true, elapsedMs: Date.now() - started, last };
      }
    } else {
      last = { status: resp.status(), body: await resp.text() };
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  return { granted: false, elapsedMs: Date.now() - started, last };
}

test("un abonnement réel ouvre l'accès via le webhook", async ({ request }) => {
  test.setTimeout(WEBHOOK_DEADLINE_MS + 120_000);

  const email = uniqueEmail();
  let subscriptionId: string | null = null;

  try {
    // 1) Un compte neuf, par la VRAIE route. Le contexte `request` garde les
    //    cookies, donc la session d'inscription sert aux appels authentifiés.
    const registered = await request.post(`${API_BASE}/api/auth/register`, {
      data: {
        email,
        password: 'pay3-live-test-password-1',
        age_confirmed: true,
        accept_terms: true,
        accept_privacy: true,
      },
    });
    expect(
      registered.status(),
      `inscription impossible (${registered.status()}) : ${await registered.text()}`,
    ).toBe(201);

    // 2) Avant de payer, l'accès doit être FERMÉ. Sans cette vérification, un
    //    test vert ne prouverait rien : un mur désactivé passerait pour un succès.
    const beforePay = await request.get(
      `${API_BASE}/api/market-status?instrument=XAUUSD&timeframe=M15`,
    );
    expect(
      beforePay.status(),
      'le mur ne mord pas avant paiement — le test ne prouverait rien. ' +
        'SUBSCRIPTION_GATE_ENFORCED=1 est-il posé sur le backend de test ?',
    ).toBe(402);

    // 3) La route de paiement crée le client Stripe ET le lie au compte. On
    //    l'appelle vraiment : c'est cette liaison que le webhook utilisera
    //    ensuite pour retrouver le compte.
    const checkout = await request.post(`${API_BASE}/api/billing/checkout`, {
      data: { plan_key: 'MONTHLY' },
    });
    expect(
      checkout.status(),
      `création de la session Checkout impossible (${checkout.status()}) : ${await checkout.text()}`,
    ).toBe(200);

    // 4) Retrouver ce client chez Stripe, par l'e-mail du compte.
    const found = await request.get(
      `${STRIPE_API}/v1/customers?email=${encodeURIComponent(email)}&limit=1`,
      { headers: { Authorization: `Bearer ${SECRET}` } },
    );
    const customers = (await found.json()) as { data?: Array<{ id: string }> };
    const customerId = customers.data?.[0]?.id;
    expect(
      customerId,
      "aucun client Stripe pour ce compte — /api/billing/checkout ne l'a pas créé",
    ).toBeTruthy();

    // 5) Un moyen de paiement de test, attaché et posé par défaut. `tok_visa`
    //    est le jeton de test Stripe correspondant à la carte 4242.
    const pm = await stripe(request, '/v1/payment_methods', {
      type: 'card',
      'card[token]': 'tok_visa',
    });
    const pmId = String(pm.id);
    await stripe(request, `/v1/payment_methods/${pmId}/attach`, { customer: customerId! });
    await stripe(request, `/v1/customers/${customerId}`, {
      'invoice_settings[default_payment_method]': pmId,
    });

    // 6) L'abonnement, facturé pour de vrai. `error_if_incomplete` fait échouer
    //    ICI si la carte ne passe pas, plutôt que de laisser un abonnement
    //    incomplet ressembler plus tard à un webhook mort.
    const sub = await stripe(request, '/v1/subscriptions', {
      customer: customerId!,
      'items[0][price]': PRICE_ID,
      payment_behavior: 'error_if_incomplete',
    });
    subscriptionId = String(sub.id);
    expect(
      String(sub.status),
      `l'abonnement Stripe n'est pas actif (${String(sub.status)}) — le paiement lui-même a échoué`,
    ).toMatch(/^(active|trialing)$/);

    // 7) LE point de mesure. Stripe a émis ses événements ; `stripe listen` les
    //    relaie ; notre webhook doit ouvrir l'accès. On ne touche pas à /sync.
    const { granted, elapsedMs, last } = await waitForWebhookAccess(request);

    expect(
      granted,
      [
        '',
        '=========================================================================',
        "PAYÉ MAIS PAS D'ACCÈS — le webhook Stripe n'a pas mis l'état à jour en " +
          `${Math.round(elapsedMs / 1000)} s.`,
        '',
        "L'abonnement est ACTIF chez Stripe : le paiement a bien eu lieu. C'est la",
        'chaîne webhook → accès qui est cassée. Un client qui vient de payer',
        'resterait enfermé dehors, sans que rien ne le signale.',
        '',
        'NE PAS désactiver ce test pour débloquer un déploiement : la panne est',
        'réelle, et coûte un client à chaque fois.',
        '',
        'À vérifier, dans cet ordre :',
        '  1. le endpoint webhook est-il joignable et pointé sur /api/billing/webhook ?',
        '  2. STRIPE_WEBHOOK_SECRET correspond-il à CE endpoint (et au bon mode) ?',
        '  3. le journal de livraison Stripe montre-t-il des 4xx/5xx ?',
        '  4. les événements customer.subscription.* sont-ils bien abonnés ?',
        '',
        `Abonnement Stripe : ${subscriptionId} (statut ${String(sub.status)})`,
        `Dernier état lu côté app : ${JSON.stringify(last)}`,
        '=========================================================================',
      ].join('\n'),
    ).toBe(true);

    // 8) Et le mur doit s'être VRAIMENT ouvert sur la route de données, pas
    //    seulement dans la réponse d'abonnement. On n'exige pas un 200 : un
    //    runner n'a pas forcément de données de marché chargées, et l'exiger
    //    rendrait le test rouge pour une raison étrangère à l'accès.
    const afterPay = await request.get(
      `${API_BASE}/api/market-status?instrument=XAUUSD&timeframe=M15`,
    );
    expect(
      [401, 402, 403],
      `l'abonnement est actif mais la route de données refuse encore l'accès ` +
        `(${afterPay.status()}) — le point de décision unique ne voit pas l'abonnement`,
    ).not.toContain(afterPay.status());
  } finally {
    // Le ménage ne doit jamais changer le verdict du test.
    if (subscriptionId) {
      await request
        .delete(`${STRIPE_API}/v1/subscriptions/${subscriptionId}`, {
          headers: { Authorization: `Bearer ${SECRET}` },
        })
        .catch(() => undefined);
    }
  }
});
