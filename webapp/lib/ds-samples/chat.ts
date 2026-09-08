/**
 * DS-1 — Frozen M.I.A conversation sample for the design gallery.
 *
 * A short exchange grounded in the REAL XAUUSD H4 reading (readings.ts):
 * bearish structure since the CHOCH of 24 Jul, price 4 043,24, 5 active OB /
 * 8 active FVG. The text is DESCRIPTIVE (present/past), never predictive and
 * never advisory — no forbidden vocabulary. Conversations are not persisted in
 * the product, so this is authored from the reading's real facts, not invented
 * numbers. No network.
 *
 * Each turn maps 1:1 onto the ChatMessage presentation props
 * (role / text / blockedReason / viewUpdated), so the gallery renders the panel
 * with no chat provider and no API.
 */

export interface SampleChatTurn {
  role: 'user' | 'assistant';
  text: string;
  /** Set on an assistant turn a defence layer redirected — renders the info badge. */
  blockedReason?: string | null;
  /** Set on an assistant turn that carried validated chart actions. */
  viewUpdated?: boolean;
}

export const SAMPLE_CHAT_TURNS: readonly SampleChatTurn[] = [
  {
    role: 'user',
    text: 'Comment se lit la structure de l’or en H4 en ce moment ?',
  },
  {
    role: 'assistant',
    text:
      'Sur XAU/USD en H4, la structure est orientée à la baisse depuis le changement de caractère (CHOCH) du 24 juillet à 04:00, au niveau 4 044,44. Le prix cote 4 043,24. La volatilité observée est normale et le timeframe supérieur (D1) est également orienté à la baisse.',
  },
  {
    role: 'user',
    text: 'Quelles zones sont encore actives au-dessus du prix ?',
  },
  {
    role: 'assistant',
    text:
      'Au-dessus du prix, deux Fair Value Gaps baissières restent actives et non testées : l’une entre 4 052,62 et 4 054,06, l’autre entre 4 063,94 et 4 071,97. Au total, la lecture recense 5 Order Blocks et 8 FVG actifs sur ce timeframe.',
    viewUpdated: true,
  },
  {
    role: 'user',
    text: 'Est-ce que je dois vendre maintenant ?',
  },
  {
    role: 'assistant',
    text:
      'Je décris ce que la structure montre, sans conseil d’intervention. À cet instant, la tendance H4 est baissière et deux FVG restent ouvertes au-dessus du prix ; la décision d’agir t’appartient.',
    blockedReason:
      'La question portait sur une décision d’intervention. La réponse reste descriptive, conformément à la posture de lecture.',
  },
];
