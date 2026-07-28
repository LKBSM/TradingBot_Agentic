import { act, fireEvent, renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import * as React from 'react';
import { ChartViewProvider, useChartView } from '../viewState';

/**
 * VZ-1 — the unified selection is the SINGLE source of truth: exactly one element
 * is selected product-wide, and the legacy `highlightZoneId` / `referenceLevel`
 * channels are DERIVED from it. These tests lock the single-selection invariant
 * and the derivations the chart/panels rely on.
 */
function wrapper({ children }: { children: React.ReactNode }) {
  return <ChartViewProvider>{children}</ChartViewProvider>;
}

describe('unified selection (VZ-1)', () => {
  it('selecting a zone drives the derived highlightZoneId', () => {
    const { result } = renderHook(() => useChartView(), { wrapper });
    act(() => result.current.select({ family: 'zone', id: 'OB_1' }));
    expect(result.current.selection).toEqual({ family: 'zone', id: 'OB_1' });
    expect(result.current.view.highlightZoneId).toBe('OB_1');
  });

  it('selecting a LEVEL deselects an active ZONE (single selection product-wide)', () => {
    const { result } = renderHook(() => useChartView(), { wrapper });
    act(() => result.current.select({ family: 'zone', id: 'OB_1' }));
    act(() =>
      result.current.select({
        family: 'level',
        kind: 'liquidity',
        id: 'LIQ_1',
        price: 2400,
        label: 'BSL · 2 400',
        side: 'bsl',
      }),
    );
    expect(result.current.selection?.family).toBe('level');
    // The zone highlight is gone — only one thing is ever selected.
    expect(result.current.view.highlightZoneId).toBeNull();
  });

  it('selecting an EVENT deselects a level and derives no highlight', () => {
    const { result } = renderHook(() => useChartView(), { wrapper });
    act(() =>
      result.current.select({
        family: 'level',
        kind: 'liquidity',
        id: 'LIQ_1',
        price: 2400,
        label: 'x',
        side: 'ssl',
      }),
    );
    act(() =>
      result.current.select({
        family: 'event',
        id: 'bos:1000:2400',
        kind: 'bos',
        direction: 'bearish',
        level: 2400,
        atSec: 1000,
      }),
    );
    expect(result.current.selection?.family).toBe('event');
    expect(result.current.view.highlightZoneId).toBeNull();
    expect(result.current.referenceLevel).toBeNull();
  });

  it('setReferenceLevel derives referenceLevel, and a zone click clears it', () => {
    const { result } = renderHook(() => useChartView(), { wrapper });
    act(() => result.current.setReferenceLevel({ price: 4202.03, label: 'Haut de la veille' }));
    expect(result.current.referenceLevel).toEqual({
      price: 4202.03,
      label: 'Haut de la veille',
    });
    expect(result.current.selection?.family).toBe('level');
    act(() => result.current.select({ family: 'zone', id: 'FVG_9' }));
    expect(result.current.referenceLevel).toBeNull();
    expect(result.current.view.highlightZoneId).toBe('FVG_9');
  });

  it('clearSelection drops the selection and all derived emphasis', () => {
    const { result } = renderHook(() => useChartView(), { wrapper });
    act(() => result.current.select({ family: 'zone', id: 'OB_1' }));
    act(() => result.current.clearSelection());
    expect(result.current.selection).toBeNull();
    expect(result.current.view.highlightZoneId).toBeNull();
  });

  it('chatbot focus_zone + highlight_zone route to the same single zone selection', () => {
    const { result } = renderHook(() => useChartView(), { wrapper });
    act(() =>
      result.current.applyActions([
        { action: 'focus_zone', params: { zone_id: 'OB_7' } },
        { action: 'highlight_zone', params: { zone_id: 'OB_7' } },
      ]),
    );
    expect(result.current.selection).toEqual({ family: 'zone', id: 'OB_7' });
    act(() => result.current.applyActions([{ action: 'clear_highlight', params: {} }]));
    expect(result.current.selection).toBeNull();
  });

  it('reset_view / reset clear the selection too', () => {
    const { result } = renderHook(() => useChartView(), { wrapper });
    act(() => result.current.select({ family: 'zone', id: 'OB_1' }));
    act(() => result.current.reset());
    expect(result.current.selection).toBeNull();
  });

  it('Escape deselects the active element (and is a no-op when nothing is selected)', () => {
    const { result } = renderHook(() => useChartView(), { wrapper });
    // No-op when nothing is selected.
    act(() => {
      fireEvent.keyDown(window, { key: 'Escape' });
    });
    expect(result.current.selection).toBeNull();
    // Deselects an active element.
    act(() => result.current.select({ family: 'zone', id: 'OB_1' }));
    act(() => {
      fireEvent.keyDown(window, { key: 'Escape' });
    });
    expect(result.current.selection).toBeNull();
  });
});
