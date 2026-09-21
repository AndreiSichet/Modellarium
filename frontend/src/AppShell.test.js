import { render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import App from './App';
import { createPrediction, getHealth, getSchedule } from './api';

jest.mock('./api');

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  getSchedule.mockResolvedValue([]);
  getHealth.mockResolvedValue({ dataAsOf: '2026-04-12', daysBehind: 161, stale: true });
  createPrediction.mockResolvedValue(null);
});

describe('app shell', () => {
  test('/ lands on About, not on predictions', () => {
    renderAt('/');

    expect(
      screen.getByRole('heading', {
        name: /Computing sports predictions for the love of the game/,
      })
    ).toBeInTheDocument();
  });

  test('the logo is hidden from assistive tech so the wordmark is not read twice', () => {
    const { container } = renderAt('/');

    const logo = container.querySelector('.wordmark-logo');
    expect(logo).toHaveAttribute('alt', '');
    expect(logo).toHaveAttribute('aria-hidden', 'true');

    const header = screen.getByRole('banner');
    expect(within(header).getAllByText(/MODELLARIUM/i)).toHaveLength(1);
  });

  test('the active link is derived from the URL, not from a click', async () => {
    const { unmount } = renderAt('/');
    expect(screen.getByRole('link', { name: 'About' })).toHaveClass('active');
    unmount();

    renderAt('/predictions/basketball');
    await waitFor(() =>
      expect(screen.getByRole('link', { name: 'Predictions' })).toHaveClass('active')
    );
    expect(screen.getByRole('link', { name: 'About' })).not.toHaveClass('active');
  });

  test('About does not stay active on a child route', async () => {
    renderAt('/predictions/basketball');

    await waitFor(() =>
      expect(screen.getByRole('link', { name: 'About' })).not.toHaveClass('active')
    );
  });

  test('Predictions stays active on a league page', async () => {
    renderAt('/predictions/basketball/nba');

    await waitFor(() =>
      expect(screen.getByRole('link', { name: 'Predictions' })).toHaveClass('active')
    );
  });
});

describe('sports rail', () => {
  test('lists only sports that exist — no placeholder entries', async () => {
    renderAt('/predictions/basketball');

    const rail = screen.getByRole('navigation', { name: 'Sports' });
    await waitFor(() => expect(within(rail).getAllByRole('link')).toHaveLength(1));
    expect(within(rail).getByRole('link', { name: /Basketball/ })).toBeInTheDocument();
  });
});

describe('about page', () => {
  test('renders all four prose blocks', () => {
    renderAt('/');

    for (const heading of [
      'How predictions are made',
      'Choosing a model',
      "What it doesn't claim",
      'Staying current',
    ]) {
      expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument();
    }
  });

  test('the retraining copy describes a regression guard, not an improvement bar', () => {
    renderAt('/');

    expect(
      screen.getByText(/only replaces the current one if it is not worse/)
    ).toBeInTheDocument();
  });
});
