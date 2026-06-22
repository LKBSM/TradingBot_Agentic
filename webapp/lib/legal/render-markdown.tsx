import * as React from 'react';

/**
 * Heading-aware, dependency-free Markdown renderer for legal documents.
 *
 * The CGU markdown (`docs/legal/conditions-utilisation.md`) is rendered TEL QUEL
 * — this turns its `#`/`##`/`###` headings, `>` blockquotes, `-` bullet lists,
 * `**bold**`/`*italic*` and paragraphs into React nodes WITHOUT ever using
 * `dangerouslySetInnerHTML`, so React escapes all text (no XSS surface even
 * though the source is trusted/server-owned).
 *
 * It is intentionally separate from the chat renderer (`lib/chat/markdown.tsx`),
 * which has no heading support — legal docs are heading-heavy.
 */

const INLINE_RE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*|_[^_\n]+_)/g;
const HEADING_RE = /^(#{1,6})\s+(.*)$/;
const BULLET_RE = /^\s*[-*•]\s+(.*)$/;
const BLOCKQUOTE_RE = /^\s*>\s?(.*)$/;
const HR_RE = /^\s*(?:-{3,}|_{3,}|\*{3,})\s*$/;

function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  text.split(INLINE_RE).forEach((part, i) => {
    if (!part) return;
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      nodes.push(<strong key={key}>{part.slice(2, -2)}</strong>);
    } else if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      nodes.push(
        <code key={key} className="rounded bg-foreground/10 px-1 py-0.5 font-mono text-[0.85em]">
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

const HEADING_CLASS: Record<number, string> = {
  1: 'text-2xl font-semibold tracking-tight sm:text-3xl mt-0',
  2: 'text-xl font-semibold tracking-tight mt-8',
  3: 'text-lg font-semibold mt-6',
  4: 'text-base font-semibold mt-4',
  5: 'text-sm font-semibold mt-4',
  6: 'text-sm font-semibold mt-4',
};

export function renderLegalMarkdown(input: string): React.ReactNode {
  const lines = input.replace(/\r\n/g, '\n').split('\n');
  const blocks: React.ReactNode[] = [];

  let para: string[] = [];
  let bullets: string[] = [];
  let quote: string[] = [];
  let blockId = 0;

  const flushPara = () => {
    if (para.length === 0) return;
    const id = `p-${blockId++}`;
    blocks.push(
      <p key={id} className="mt-3 leading-relaxed text-muted-foreground">
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
      <ul key={id} className="mt-3 list-disc space-y-1 pl-6 text-muted-foreground">
        {bullets.map((item, i) => (
          <li key={`${id}-i-${i}`}>{renderInline(item, `${id}-i-${i}`)}</li>
        ))}
      </ul>,
    );
    bullets = [];
  };

  const flushQuote = () => {
    if (quote.length === 0) return;
    const id = `bq-${blockId++}`;
    blocks.push(
      <blockquote
        key={id}
        className="mt-4 border-l-2 border-border pl-4 text-sm italic text-muted-foreground"
      >
        {quote.map((line, i) => (
          <React.Fragment key={`${id}-l-${i}`}>
            {i > 0 && <br />}
            {renderInline(line, `${id}-l-${i}`)}
          </React.Fragment>
        ))}
      </blockquote>,
    );
    quote = [];
  };

  const flushAll = () => {
    flushPara();
    flushBullets();
    flushQuote();
  };

  for (const line of lines) {
    if (line.trim() === '') {
      flushAll();
      continue;
    }
    if (HR_RE.test(line)) {
      flushAll();
      blocks.push(<hr key={`hr-${blockId++}`} className="my-6 border-border/60" />);
      continue;
    }
    const heading = line.match(HEADING_RE);
    if (heading) {
      flushAll();
      const level = heading[1]!.length;
      const text = heading[2]!;
      const cls = HEADING_CLASS[level] ?? HEADING_CLASS[6];
      const Tag = `h${Math.min(level, 6)}` as keyof React.JSX.IntrinsicElements;
      blocks.push(
        <Tag key={`h-${blockId++}`} className={`${cls} text-foreground`}>
          {renderInline(text, `h-${blockId}`)}
        </Tag>,
      );
      continue;
    }
    const bq = line.match(BLOCKQUOTE_RE);
    if (bq) {
      flushPara();
      flushBullets();
      quote.push(bq[1]!);
      continue;
    }
    const bullet = line.match(BULLET_RE);
    if (bullet) {
      flushPara();
      flushQuote();
      bullets.push(bullet[1]!);
      continue;
    }
    flushBullets();
    flushQuote();
    para.push(line);
  }
  flushAll();

  return <>{blocks}</>;
}
