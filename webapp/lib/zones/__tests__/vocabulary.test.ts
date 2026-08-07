import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';

/**
 * VZ-1 vocabulary discipline (mission §0 / §5), enforced on the SHIPPED strings.
 * The zones namespace must never say « chevauche » / « overlap », never pass a
 * value judgement on a zone, never show a distance without its unit + reference
 * edge, and never invite the user to relax a filter.
 */

function strings(obj: unknown, out: string[] = []): string[] {
  if (typeof obj === 'string') out.push(obj);
  else if (obj && typeof obj === 'object') for (const v of Object.values(obj)) strings(v, out);
  return out;
}

const FR = strings((fr as Record<string, unknown>).zones).map((s) => s.toLowerCase());
const EN = strings((en as Record<string, unknown>).zones).map((s) => s.toLowerCase());

describe('« chevauche » / « overlap » are banished', () => {
  it('never in the French zones strings', () => {
    expect(FR.filter((s) => s.includes('chevauche'))).toEqual([]);
  });
  it('never in the English zones strings', () => {
    expect(EN.filter((s) => s.includes('overlap'))).toEqual([]);
  });
});

describe('no value judgement on a zone', () => {
  // These stems assert a zone is good/valid/respected/strong/reliable/quality/best.
  const FR_BAD = ['respect', 'valid', 'solide', 'fiable', 'qualité', 'meilleur', ' forte', ' fort '];
  const EN_BAD = ['respect', 'valid', 'solid', 'strong', 'reliab', 'quality', 'better', 'best'];
  it('French', () => {
    for (const bad of FR_BAD) expect(FR.filter((s) => s.includes(bad)), bad).toEqual([]);
  });
  it('English', () => {
    for (const bad of EN_BAD) expect(EN.filter((s) => s.includes(bad)), bad).toEqual([]);
  });
});

describe('a distance always carries its unit AND its reference edge', () => {
  it('the outside-distance template', () => {
    const line = (fr as { zones: { proximity: { distanceLine: string } } }).zones.proximity.distanceLine;
    expect(line).toContain('pts');
    expect(line).toContain('{edge}');
    expect(line).toContain('{side}');
  });
  it('the inside template names both edges', () => {
    const line = (fr as { zones: { proximity: { insideDist: string } } }).zones.proximity.insideDist;
    expect(line).toContain('pts');
    expect(line).toContain('bord bas');
    expect(line).toContain('bord haut');
  });
});

describe('an empty filter never suggests relaxing / broadening', () => {
  const RELAX = ['assoupl', 'élargir', 'elargir', 'relax', 'broaden', 'moins strict', 'try a'];
  it('French + English empty-filter messages', () => {
    const msgs = [...Object.values((fr as { zones: { emptyFilter: Record<string, string> } }).zones.emptyFilter),
      ...Object.values((en as { zones: { emptyFilter: Record<string, string> } }).zones.emptyFilter)].map((s) => s.toLowerCase());
    for (const bad of RELAX) expect(msgs.filter((s) => s.includes(bad)), bad).toEqual([]);
  });
});

describe('the honesty note explains the no-judgement choice (mission §D)', () => {
  it('exists in both languages', () => {
    expect((fr as { zones: { contacts: { honesty: string } } }).zones.contacts.honesty.length).toBeGreaterThan(40);
    expect((en as { zones: { contacts: { honesty: string } } }).zones.contacts.honesty.length).toBeGreaterThan(40);
  });
});
