import '@testing-library/jest-dom/vitest';
import { configure } from '@testing-library/dom';

// The jsdom environment can be slow under parallel file execution on some
// machines; give async utilities (waitFor / findBy*) a larger budget so
// timing-sensitive tests don't flake under load.
configure({ asyncUtilTimeout: 5000 });
