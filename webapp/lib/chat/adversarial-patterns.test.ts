import { describe, expect, it } from 'vitest';
import {
  ADVERSARIAL_PATTERNS_EN,
  ADVERSARIAL_PATTERNS_FR,
  BENIGN_PATTERNS,
  classifyUserInput,
} from './adversarial-patterns';

describe('DG-112 adversarial catalogue — recall ≥ 98 %', () => {
  it('FR catalogue has at least 30 patterns', () => {
    expect(ADVERSARIAL_PATTERNS_FR.length).toBeGreaterThanOrEqual(30);
  });

  it('EN catalogue has at least 18 patterns', () => {
    expect(ADVERSARIAL_PATTERNS_EN.length).toBeGreaterThanOrEqual(18);
  });

  it('every FR pattern fires the gate with a non-null category', () => {
    const misses: string[] = [];
    for (const p of ADVERSARIAL_PATTERNS_FR) {
      const m = classifyUserInput(p.input, 'fr');
      if (m === null) misses.push(p.input);
    }
    const recall = 1 - misses.length / ADVERSARIAL_PATTERNS_FR.length;
    expect(recall, `Missed: ${misses.join(' | ')}`).toBeGreaterThanOrEqual(0.98);
  });

  it('every EN pattern fires the gate with a non-null category', () => {
    const misses: string[] = [];
    for (const p of ADVERSARIAL_PATTERNS_EN) {
      const m = classifyUserInput(p.input, 'en');
      if (m === null) misses.push(p.input);
    }
    const recall = 1 - misses.length / ADVERSARIAL_PATTERNS_EN.length;
    expect(recall, `Missed: ${misses.join(' | ')}`).toBeGreaterThanOrEqual(0.98);
  });

  it('cross-language pass: FR pattern still trips when label-hint is EN', () => {
    // Defence-in-depth: a user who types French but the route guessed EN
    // should still get blocked (or vice versa).
    const m = classifyUserInput('Achetez ou vendez ?', 'en');
    expect(m).not.toBeNull();
  });

  it('classification preserves the expected category for at least 90 % of FR patterns', () => {
    let correct = 0;
    for (const p of ADVERSARIAL_PATTERNS_FR) {
      const m = classifyUserInput(p.input, 'fr');
      if (m && m.category === p.expected_category) correct += 1;
    }
    const accuracy = correct / ADVERSARIAL_PATTERNS_FR.length;
    expect(accuracy).toBeGreaterThanOrEqual(0.9);
  });

  it('classification preserves the expected category for at least 90 % of EN patterns', () => {
    let correct = 0;
    for (const p of ADVERSARIAL_PATTERNS_EN) {
      const m = classifyUserInput(p.input, 'en');
      if (m && m.category === p.expected_category) correct += 1;
    }
    const accuracy = correct / ADVERSARIAL_PATTERNS_EN.length;
    expect(accuracy).toBeGreaterThanOrEqual(0.9);
  });
});

describe('DG-112 benign control set — false-positive rate < 5 %', () => {
  it('benign questions do NOT trigger the gate', () => {
    const falsePositives: string[] = [];
    for (const p of BENIGN_PATTERNS) {
      const m = classifyUserInput(p.input, p.language);
      if (m !== null) falsePositives.push(`${p.input} → ${m.category}`);
    }
    const fpRate = falsePositives.length / BENIGN_PATTERNS.length;
    expect(
      fpRate,
      `False positives: ${falsePositives.join(' | ')}`,
    ).toBeLessThan(0.05);
  });
});

describe('DG-112 edge cases', () => {
  it('empty input → null', () => {
    expect(classifyUserInput('')).toBeNull();
    expect(classifyUserInput('   ')).toBeNull();
  });

  it('whitespace-only input → null', () => {
    expect(classifyUserInput('\n\n\t')).toBeNull();
  });

  it('language auto-detection: French particles trip FR rules', () => {
    const m = classifyUserInput("Dois-je acheter maintenant ?");
    expect(m?.language).toBe('fr');
  });

  it('language auto-detection: pure English trips EN rules', () => {
    const m = classifyUserInput('Should I buy now?');
    expect(m?.language).toBe('en');
  });
});
