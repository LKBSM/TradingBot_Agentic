import { test, expect, type APIRequestContext, type Page } from '@playwright/test';

/**
 * PAY-3 (G1) — LE TEST VIVANT. La défense contre le scénario le plus coûteux.
 *
 * Le scénario : le webhook cesse de mettre à jour l'état, le client paie, Stripe
 * est content, l'app le croit non abonné, personne ne le sait. Ça ne déclenche
 * aucune alarme ; ça se lit comme une baisse de conversion qu'on impute à autre
 * chose pendant des semaines.
 *
 * Aucun test à faux client Stripe ne peut attraper ça — c'est précisément ainsi
 * qu'un `verify_webhook` renvoyant un objet `stripe.Event` (dont `.get()` lève)
 * est passé en production : le faux client, lui, renvoyait un `dict`. Donc ce
 * test-ci pousse une VRAIE carte de test à travers le VRAI Checkout hébergé,
 * attend le VRAI webhook, et échoue bruyamment si l'accès ne s'ouvre pas.
 *
 * CE QU'IL NE FAIT SURTOUT PAS
 * ----------------------------
 * Il n'appelle JAMAIS `POST /api/billing/sync`. La réconciliation directe
 * (PAY-3e) sauverait le client et ferait passer le test au vert alors même que
 * le webhook est mort — c'est-à-dire qu'elle masquerait exactement la panne que
 * ce test existe pour détecter. Seul le chemin webhook est mesuré ici.
 *
 * Il ne se désactive pas non plus pour faire passer un déploiement : quand
 * PAY3_LIVE_REQUIRED=1 (ce que pose le workflow), une configuration manquante
 * est un ÉCHEC, pas un skip.
 *
 * PRÉREQUIS (clés de TEST uniquement)
 * -----------------------------------
 *   STRIPE_SECRET_KEY   sk_test_… (une clé live fait échouer le test exprès)
 *   PAY3_API_BASE       base du backend, ex. http://localhost:8000
 *   PAY3_APP_BASE       base du front, ex. http://localhost:3000
 *   et, côté backend lancé pour le test :
 *     STRIPE_WEBHOOK_SECRET issu de `stripe listen`, SUBSCRIPTION_GATE_ENFORCED=1,
 *     EMAIL_VERIFICATION_ENFORCED=0 (pas de SMTP dans un runner).
 */

// Combien de temps on accorde au webhook avant de considérer que le client est
// enfermé dehors. Généreux pour un runner CI, mais fini : au-delà, un vrai
// client aurait abandonné.
const WEBHOOK_DEADLINE_MS = 90_000;
const POLL_INTERVAL_MS = 2_000;

const REQUIRED = process.env.PAY3_LIVE_REQUIRED === '1';
const SECRET = process.env.STRIPE_SECRET_KEY ?? '';
const API_BASE = (process.env.PAY3_API_BASE ?? '').replace(/\/$/, '');
const APP_BASE = (process.env.PAY3_APP_BASE ?? '').replace(/\/$/, '');

const missing: string[] = [];
if (!SECRET) missing.push('STRIPE_SECRET_KEY');
if (!API_BASE) missing.push('PAY3_API_BASE');
// PAY3_APP_BASE n'est PAS exigé : ce test ne visite jamais notre front — il va
// sur la page Checkout hébergée par Stripe et parle au backend en HTTP. Exiger
// le front obligerait le runner à construire le webapp pour rien.
void APP_BASE;

test.beforeAll(() => {
  // Un garde-fou avant tout le reste : ce test crée de vrais objets Stripe et
  // pousse un vrai numéro de carte. Sur une clé live, ce serait un débit réel.
  if (SECRET.startsWith('sk_live_')) {
    throw new Error(
      'PAY-3 live test refusé : STRIPE_SECRET_KEY est une clé LIVE. ' +
        "Ce test pousse une carte de test à travers Checkout et ne doit JAMAIS " +
        "toucher un compte réel. Utilise une clé sk_test_.",
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

/** Remplit la page Checkout HÉBERGÉE par Stripe avec la carte de test. */
async function payWithTestCard(page: Page): Promise<void> {
  // Les champs de Checkout vivent dans la page elle-même (plus d'iframe depuis
  // 2022). On attend le numéro de carte : c'est le signal que la page est prête.
  const cardNumber = page.locator('#cardNumber');
  await expect(
    cardNumber,
    "la page Checkout hébergée ne présente pas de champ carte — l'URL de session " +
      'est peut-être invalide, ou le prix Stripe est dans un autre mode (test/live)',
  ).toBeVisible({ timeout: 30_000 });

  await cardNumber.fill('4242424242424242');
  await page.locator('#cardExpiry').fill('12' + String(new Date().getFullYear() + 2).slice(-2));
  await page.locator('#cardCvc').fill('123');

  const name = page.locator('#billingName');
  if (await name.count()) await name.fill('PAY3 Test Buyer');

  // billing_address_collection=required (G2) : le pays est demandé. On paie
  // depuis le Canada, donc DANS la zone — le filet hors zone ne doit pas mordre.
  const country = page.locator('#billingCountry');
  if (await country.count()) await country.selectOption('CA');
  const postal = page.locator('#billingPostalCode');
  if (await postal.count()) await postal.fill('H2X 1Y4');

  // DÉCLARATION D'AGENT AUTOMATISÉ. Stripe Checkout présente une case
  // « I am an AI agent acting on behalf of someone else », vue sur la trace d'un
  // échec : le formulaire restait sans effet et Checkout n'affichait aucune
  // erreur. On la coche parce que c'est vrai — ce test EST un agent qui paie
  // pour le compte de quelqu'un — et parce que c'est la voie que Stripe prévoit
  // pour un paiement automatisé. Conditionnel : la case peut disparaître ou
  // changer de libellé sans que le test doive casser.
  const agentBox = page.getByRole('checkbox', {
    name: /ai agent acting on behalf|agent (ia|ai) agissant/i,
  });
  if (await agentBox.count()) {
    // `force` : Stripe rend un <input> reel masque hors viewport, pilote par un
    // libelle stylise. Sans force, Playwright refuse de cliquer ("element is
    // outside of the viewport") et mange tout le budget du test. Le timeout
    // court evite qu'un echec ici masque le vrai verdict : si la case ne se
    // coche pas, on laisse le controle de soumission ci-dessous le dire.
    await agentBox
      .first()
      .check({ force: true, timeout: 10_000 })
      .catch(() => undefined);
  }

  // LE VRAI BOUTON D'ENVOI. Piège vérifié sur une trace d'échec : les boutons de
  // portefeuille (« Apple Pay », « Payer avec Link ») sont AVANT dans le DOM, et
  // un sélecteur CSS à virgules se résout dans l'ordre du DOM, pas dans l'ordre
  // écrit. `.SubmitButton, button[type="submit"]` suivi de `.first()` cliquait
  // donc un portefeuille : le formulaire n'était jamais soumis, Checkout
  // n'affichait aucune erreur, et le test accusait le webhook à tort.
  // On vise le bouton par son libellé réel (Checkout est en français ici).
  const byLabel = page.getByRole('button', {
    name: /s['’]abonner|subscribe|payer maintenant|pay now/i,
  });
  const submit = (await byLabel.count()) ? byLabel.last() : page.locator('.SubmitButton').last();
  await expect(
    submit,
    "aucun bouton d'envoi trouvé sur la page Checkout — le libellé du bouton a " +
      'peut-être changé, ou la page a été rendue dans une autre langue',
  ).toBeVisible({ timeout: 15_000 });
  await submit.click();

  // VÉRIFIER QUE LE PAIEMENT A ABOUTI, avant d'attendre quoi que ce soit.
  // Sans ce contrôle, un clic qui échoue est INDISCERNABLE d'un webhook mort :
  // les deux donnent « pas d'accès après 90 s », et on cherche la panne au
  // mauvais endroit pendant ce temps. Checkout quitte son domaine dès que le
  // paiement passe (redirection vers success_url) — c'est le signal le plus
  // fiable, et il ne dépend pas de notre front, qui ne tourne pas forcément.
  const left = await page
    .waitForURL((u) => !u.host.endsWith('stripe.com'), { timeout: 60_000 })
    .then(() => true)
    .catch(() => false);

  if (!left) {
    const shown = await page
      .locator('[role="alert"], .FieldError, .Notice, .ConfirmPayment-Error')
      .allInnerTexts()
      .catch(() => [] as string[]);
    const visible = shown.map((t) => t.trim()).filter(Boolean).join(' | ');
    throw new Error(
      [
        '',
        '=========================================================================',
        "LE PAIEMENT N'EST PAS PASSÉ — on est resté sur la page Checkout après 60 s.",
        '',
        "Ce n'est PAS un webhook mort : Stripe n'a rien eu à annoncer. Cherche du",
        'côté du formulaire, pas du côté de la livraison des événements.',
        '',
        'À vérifier :',
        '  1. le prix STRIPE_PRICE_MONTHLY est-il dans le MÊME mode que la clé ?',
        '  2. une règle Radar bloque-t-elle la carte (dont la règle de zone CA/US) ?',
        '  3. Checkout demande-t-il un champ que le test ne remplit pas ?',
        '',
        visible
          ? 'Message affiché par Checkout : ' + visible
          : "Checkout n'affiche aucun message d'erreur.",
        "URL au moment de l'abandon : " + page.url(),
        '=========================================================================',
      ].join('\n'),
    );
  }
}

/**
 * Interroge l'état d'abonnement jusqu'à ce que l'accès s'ouvre — PAR LE WEBHOOK.
 * Retourne le dernier corps lu, et le temps écoulé.
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

/** Annule au Stripe réel ce que le test a créé (mode test, mais on ne laisse pas traîner). */
async function cleanUp(api: APIRequestContext, subscriptionId: string | null): Promise<void> {
  if (!subscriptionId) return;
  try {
    await api.delete(`https://api.stripe.com/v1/subscriptions/${subscriptionId}`, {
      headers: { Authorization: `Bearer ${SECRET}` },
    });
  } catch {
    /* le ménage ne doit jamais faire échouer le verdict du test */
  }
}

test('un paiement réel par carte de test ouvre l\'accès via le webhook', async ({ page, request }) => {
  test.setTimeout(WEBHOOK_DEADLINE_MS + 120_000);

  const email = uniqueEmail();
  const password = 'pay3-live-test-password-1';

  // 1) Un compte neuf. Le contexte `request` de Playwright garde les cookies,
  //    donc la session d'inscription sert ensuite aux appels authentifiés.
  const registered = await request.post(`${API_BASE}/api/auth/register`, {
    data: {
      email,
      password,
      age_confirmed: true,
      accept_terms: true,
      accept_privacy: true,
    },
  });
  expect(
    registered.status(),
    `inscription impossible (${registered.status()}) : ${await registered.text()}`,
  ).toBe(201);

  // 2) Avant de payer, l'accès doit être FERMÉ. Sans cette vérification, un test
  //    vert ne prouverait rien : un mur désactivé passerait pour un succès.
  const beforePay = await request.get(`${API_BASE}/api/market-status?instrument=XAUUSD&timeframe=M15`);
  expect(
    beforePay.status(),
    'le mur ne mord pas avant paiement — le test ne prouverait rien. ' +
      'SUBSCRIPTION_GATE_ENFORCED=1 est-il posé sur le backend de test ?',
  ).toBe(402);

  // 3) Checkout hébergé.
  const checkout = await request.post(`${API_BASE}/api/billing/checkout`, {
    data: { plan_key: 'MONTHLY' },
  });
  expect(
    checkout.status(),
    `création de la session Checkout impossible (${checkout.status()}) : ${await checkout.text()}`,
  ).toBe(200);
  const { url } = (await checkout.json()) as { url: string };
  expect(url, "Checkout n'a pas renvoyé d'URL").toContain('stripe.com');

  // 4) La vraie carte de test, sur la vraie page hébergée.
  await page.goto(url);
  await payWithTestCard(page);

  // 5) LE point de mesure. On n'appelle pas /sync : c'est le webhook qu'on teste.
  const { granted, elapsedMs, last } = await waitForWebhookAccess(request);

  let subscriptionId: string | null = null;
  if (last && typeof last === 'object') {
    subscriptionId = (last as { stripe_subscription_id?: string }).stripe_subscription_id ?? null;
  }
  await cleanUp(request, subscriptionId);

  expect(
    granted,
    [
      '',
      '=========================================================================',
      'PAYÉ MAIS PAS D\'ACCÈS — le webhook Stripe n\'a pas mis l\'état à jour en ' +
        `${Math.round(elapsedMs / 1000)} s.`,
      '',
      'Un client qui vient de payer resterait enfermé dehors, sans que rien ne',
      'le signale. NE PAS désactiver ce test pour débloquer un déploiement :',
      'c\'est la panne elle-même qu\'il faut corriger.',
      '',
      'À vérifier, dans cet ordre :',
      '  1. le endpoint webhook est-il joignable et pointé sur /api/billing/webhook ?',
      '  2. STRIPE_WEBHOOK_SECRET correspond-il à CE endpoint (et au bon mode) ?',
      '  3. le journal de livraison Stripe montre-t-il des 4xx/5xx ?',
      '  4. les événements customer.subscription.* sont-ils bien abonnés ?',
      '',
      `Dernier état lu : ${JSON.stringify(last)}`,
      '=========================================================================',
    ].join('\n'),
  ).toBe(true);

  // 6) Et le mur doit s'être VRAIMENT ouvert sur la route de données, pas
  //    seulement dans la réponse d'abonnement. On n'exige pas un 200 : un
  //    runner CI n'a pas forcément de données de marché chargées, et exiger 200
  //    rendrait le test rouge pour une raison qui n'a rien à voir avec l'accès.
  //    Ce qui compte est qu'aucun refus d'ACCÈS ne subsiste.
  const afterPay = await request.get(`${API_BASE}/api/market-status?instrument=XAUUSD&timeframe=M15`);
  expect(
    [401, 402, 403],
    `l'abonnement est actif mais la route de données refuse encore l'accès ` +
      `(${afterPay.status()}) — le point de décision unique ne voit pas l'abonnement`,
  ).not.toContain(afterPay.status());
});
