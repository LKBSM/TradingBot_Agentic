import * as React from 'react';

/**
 * Minimal, dependency-free Markdown renderer for chat bubbles.
 *
 * Why not react-markdown? The chatbot only ever produces light Markdown
 * (paragraphs, bullet / numbered lists, **bold**, *italic*, `code`). Pulling a
 * full CommonMark engine + sanitiser for that is overkill and a real bundle
 * cost. This renderer parses that small subset and returns React elements — it
 * NEVER uses `dangerouslySetInnerHTML`, so React escapes all text and there is
 * no XSS surface.
 *
 * Founder eval (2026-06-08): assistant replies showed raw `**asterisks**` and
 * ugly list markers because ChatMessage printed the raw string. This turns that
 * into clean prose without changing the model's posture (niveau 1.5).
 */

// ─── Inline: **bold**, *italic* / _italic_, `code` ───────────────────────────

const INLINE_PATTERN =
  /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*|_[^_\n]+_)/g;

/** Parse inline emphasis inside a single line into React nodes. */
function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const parts = text.split(INLINE_PATTERN);
  parts.forEach((part, i) => {
    if (!part) return;
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      nodes.push(<strong key={key}>{part.slice(2, -2)}</strong>);
    } else if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      nodes.push(
        <code
          key={key}
          className="rounded bg-foreground/10 px-1 py-0.5 font-mono text-[0.85em]"
        >
          {part.slice(1, -1)}
        </code>,
      );
    } else if (
      part.length > 2 &&
      ((part.startsWith('*') && part.endsWith('*')) ||
        (part.startsWith('_') && part.endsWith('_')))
    ) {
      nodes.push(<em key={key}>{part.slice(1, -1)}</em>);
    } else {
      nodes.push(<React.Fragment key={key}>{part}</React.Fragment>);
    }
  });
  return nodes;
}

// ─── Block parsing: paragraphs + unordered / ordered lists ───────────────────

const BULLET_RE = /^\s*[-*•]\s+(.*)$/;
const ORDERED_RE = /^\s*\d+[.)]\s+(.*)$/;

/**
 * Render a light-Markdown string as clean React nodes (paragraphs + lists).
 * Blank lines separate blocks; consecutive `- ` / `* ` / `• ` lines become a
 * bullet list, consecutive `1.` lines an ordered list. Single newlines inside a
 * paragraph are preserved as soft breaks.
 */
export function renderMarkdown(input: string): React.ReactNode {
  const lines = input.replace(/\r\n/g, '\n').split('\n');
  const blocks: React.ReactNode[] = [];

  let para: string[] = [];
  let bullets: string[] = [];
  let ordered: string[] = [];
  let blockId = 0;

  const flushPara = () => {
    if (para.length === 0) return;
    const id = `p-${blockId++}`;
    blocks.push(
      <p key={id} className="whitespace-pre-wrap [&:not(:first-child)]:mt-2">
        {para.map((line, i) => (
          <React.Fragment key={`${id}-l-${i}`}>
            {i > 0 && <br />}
            {renderInline(line, `${id}-l-${i}`)}
          </React.Fragment>
        ))}
      </p>,
    );
    para = [];
  };

  const flushBullets = () => {
    if (bullets.length === 0) return;
    const id = `ul-${blockId++}`;
    blocks.push(
      <ul
        key={id}
        className="list-disc space-y-1 pl-5 [&:not(:first-child)]:mt-2"
      >
        {bullets.map((item, i) => (
          <li key={`${id}-i-${i}`}>{renderInline(item, `${id}-i-${i}`)}</li>
        ))}
      </ul>,
    );
    bullets = [];
  };

  const flushOrdered = () => {
    if (ordered.length === 0) return;
    const id = `ol-${blockId++}`;
    blocks.push(
      <ol
        key={id}
        className="list-decimal space-y-1 pl-5 [&:not(:first-child)]:mt-2"
      >
        {ordered.map((item, i) => (
          <li key={`${id}-i-${i}`}>{renderInline(item, `${id}-i-${i}`)}</li>
        ))}
      </ol>,
    );
    ordered = [];
  };

  for (const line of lines) {
    if (line.trim() === '') {
      flushPara();
      flushBullets();
      flushOrdered();
      continue;
    }
    const bullet = line.match(BULLET_RE);
    if (bullet) {
      flushPara();
      flushOrdered();
      bullets.push(bullet[1]!);
      continue;
    }
    const num = line.match(ORDERED_RE);
    if (num) {
      flushPara();
      flushBullets();
      ordered.push(num[1]!);
      continue;
    }
    // Plain text line — part of the current paragraph.
    flushBullets();
    flushOrdered();
    para.push(line);
  }
  flushPara();
  flushBullets();
  flushOrdered();

  return <>{blocks}</>;
}
