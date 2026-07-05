import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { StrategyPanel } from '../StrategyPanel';
import {
  CURRENT_STRATEGY_SCHEMA_VERSION,
  type SavedStrategy,
} from '@/lib/conditions/strategy-store';

function makeStrategy(overrides: Partial<SavedStrategy> = {}): SavedStrategy {
  return {
    id: 'id-1',
    name: 'London sweep M15',
    schema_version: CURRENT_STRATEGY_SCHEMA_VERSION,
    config: { logic: 'AND', conditions: [{ type: 'mtf_aligned', direction: 'bullish' }] },
    createdAt: 1_700_000_000_000,
    lastUsedAt: 1_700_000_000_000,
    ...overrides,
  };
}

const noopHandlers = {
  onLoad: vi.fn(),
  onRename: vi.fn().mockReturnValue({ ok: true, strategy: makeStrategy() }),
  onDuplicate: vi.fn().mockReturnValue({ ok: true, strategy: makeStrategy() }),
  onDelete: vi.fn(),
};

describe('StrategyPanel', () => {
  it('renders nothing when there is no saved strategy', () => {
    const { container } = render(
      <StrategyPanel strategies={[]} locale="fr" {...noopHandlers} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('shows a valid strategy with an enabled Charger button and loads it', () => {
    const onLoad = vi.fn();
    const strategy = makeStrategy();
    render(
      <StrategyPanel
        strategies={[strategy]}
        locale="fr"
        {...noopHandlers}
        onLoad={onLoad}
      />,
    );
    expect(screen.getByText('London sweep M15')).toBeInTheDocument();
    const load = screen.getByRole('button', { name: 'Charger' });
    expect(load).toBeEnabled();
    fireEvent.click(load);
    expect(onLoad).toHaveBeenCalledWith(strategy);
  });

  it('marks an out-of-schema strategy invalid, shows the reasons and disables loading', () => {
    const onLoad = vi.fn();
    const stale = makeStrategy({
      name: 'Vieille stratégie',
      config: {
        logic: 'AND',
        conditions: [{ type: 'per_tf_trend_is' } as never],
      },
    });
    render(
      <StrategyPanel
        strategies={[stale]}
        locale="fr"
        {...noopHandlers}
        onLoad={onLoad}
      />,
    );
    expect(screen.getByText('Invalide')).toBeInTheDocument();
    expect(
      screen.getByText(/Condition non reconnue : « per_tf_trend_is »/),
    ).toBeInTheDocument();
    const load = screen.getByRole('button', { name: 'Charger' });
    expect(load).toBeDisabled();
    fireEvent.click(load);
    expect(onLoad).not.toHaveBeenCalled();
  });

  it('requires an explicit confirmation before deleting', () => {
    const onDelete = vi.fn();
    render(
      <StrategyPanel
        strategies={[makeStrategy()]}
        locale="fr"
        {...noopHandlers}
        onDelete={onDelete}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Supprimer' }));
    expect(onDelete).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer la suppression' }));
    expect(onDelete).toHaveBeenCalledWith('id-1');
  });

  it('renames through the inline editor', () => {
    const onRename = vi.fn().mockReturnValue({
      ok: true,
      strategy: makeStrategy({ name: 'Continuation H4' }),
    });
    render(
      <StrategyPanel
        strategies={[makeStrategy()]}
        locale="fr"
        {...noopHandlers}
        onRename={onRename}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Renommer' }));
    const input = screen.getByLabelText('Nouveau nom');
    fireEvent.change(input, { target: { value: 'Continuation H4' } });
    fireEvent.click(screen.getByRole('button', { name: 'OK' }));
    expect(onRename).toHaveBeenCalledWith('id-1', 'Continuation H4');
  });

  it('surfaces an honest error when a mutation is refused (cap reached)', () => {
    const onDuplicate = vi
      .fn()
      .mockReturnValue({ ok: false, error: 'limit_reached' });
    render(
      <StrategyPanel
        strategies={[makeStrategy()]}
        locale="fr"
        {...noopHandlers}
        onDuplicate={onDuplicate}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Dupliquer' }));
    expect(screen.getByRole('alert').textContent).toMatch(/Limite de stratégies atteinte/);
  });
});
