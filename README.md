# Market Advisor

A daily research brief in your browser. Three times a day it pulls real prices and the latest news for a list of stocks and cryptocurrencies, works out the trend numbers, and has an AI weigh the news and the numbers together. It then publishes:

- **Long-term stock ideas** to hold for months, each with a stop-loss, bullet-point reasons and risks
- **Short-term crypto ideas** to hold for days, with stop, target and time limit
- **A review of your holdings**: hold, sell, trim or add, and why
- **Alerts** when a stop-loss is hit, a target is reached, a trend breaks or bad news lands (emailed to you via GitHub)
- **News flashes** that could change a decision
- **A to-do list** for the day

It never places trades. It's research for your own decisions, not personalised financial advice.

It runs for free: GitHub runs it, hosts the website, and provides the AI model. Price and news data come from Alpaca's free market-data feed (a paper account's keys are all it needs).

---

## One-time setup (about 20 minutes)

You need: a GitHub account, and your Alpaca paper API key and secret (the same ones you already have).

### 1. Create the repository

1. github.com → **+** (top right) → **New repository**
2. Name: `market-advisor`
3. Choose **Public** (the free website hosting needs it; only the briefs are visible, never your keys)
4. Don't tick anything else → **Create repository**

### 2. Upload the files

1. Click the **uploading an existing file** link (or the address bar trick: add `/upload/main` to the repository address)
2. Unzip `market-advisor.zip`. Drag in **everything inside the folder**, including the `docs` and `.github` folders
3. **Commit changes**

Then check the file list shows `docs` and `.github`. If either is missing:

- `docs/index.html`: **Add file → Create new file**, name it `docs/index.html`, paste in the contents of that file (open it in Notepad, Ctrl+A, Ctrl+C), commit.
- `.github/workflows/advisor.yml`: same again with that name and that file's contents.

### 3. Add your Alpaca keys

**Settings → Secrets and variables → Actions → New repository secret**, twice:

| Name | Secret |
|---|---|
| `ALPACA_API_KEY_ID` | your Alpaca paper API Key ID |
| `ALPACA_API_SECRET_KEY` | your Alpaca paper Secret Key |

### 4. Let the workflow write to the repository

**Settings → Actions → General → Workflow permissions → Read and write permissions → Save.**

### 5. Turn on the website

**Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main`, folder: `/docs` → Save.**

GitHub shows your address at the top of that page, like `https://yourname.github.io/market-advisor/`. Bookmark it on your phone too.

### 6. Run it once

**Actions → Market advisor → Run workflow → Run workflow.** About two minutes later, refresh your website address. The first brief is there.

If the Actions run shows a red cross, click into it and read the line starting **Problem:** — it says what to fix.

---

## Every day after that

| Run | UK time (summer) | What it's for |
|---|---|---|
| Morning | 07:45 | The main brief before the US open (every day, crypto included) |
| Midday | 16:30 | One hour into the US session: alerts on holdings, fresh news |
| Evening | 20:45 | Fifteen minutes before the close: final check |

Alerts marked **high** (stop-loss hit, target reached, market down 2%+) also open an **issue** in your repository, and GitHub emails you. Close the issue once you've dealt with it. Install the GitHub mobile app if you want them as phone notifications.

## Tell it what you own

Open `portfolio.json` in your repository (pencil icon), add a line per holding, commit:

```json
{"symbol": "RTX", "sleeve": "long", "qty": 10, "entry_price": 152.40, "entry_date": "2026-09-16"}
{"symbol": "BTC/USD", "sleeve": "short", "qty": 0.05, "entry_price": 76500, "entry_date": "2026-09-17"}
```

`sleeve` is `long` (stocks, months) or `short` (crypto, days). Optional per line: `stop_pct`, `target_pct`, `note`. The next brief reviews each one and alerts you when something needs doing. The repository is public, so leave `qty` out if you'd rather not show sizes — the advice still works from the entry price.

## Change what it watches

`config.toml` has the two lists (`[long]` stocks, `[short]` crypto), the stop and target percentages, and how many ideas to show. `events.txt` is a plain list of upcoming events (Fed meetings, earnings, geopolitics) the AI is told to keep in mind — update it when something big is coming.

## The AI

By default it uses GitHub's free model access (GPT-4.1) through the workflow's own token, so there's nothing to sign up for. Each run makes three requests; the free allowance is 50 a day.

If you'd rather use Claude, add a secret named `ANTHROPIC_API_KEY` and it switches automatically. That's a pay-as-you-go key from console.anthropic.com, separate from a Claude subscription.

The AI can only rate symbols it's given. Every price, stop and target is checked against the real numbers, and the rules keep the final word on your holdings: if a stop-loss has been hit, the advice is **sell** whatever the AI thinks. If the AI is unavailable, a rules-only brief is published instead and says so.

## Good to know

- The briefs are research, not instructions. The AI can misread a headline and the rules are simple. Read the reasons, check the news link, and size positions sensibly.
- Crypto moves fast and trades at weekends; the morning run covers it every day.
- The US market clock comes from Alpaca, so holidays are handled.
- Past briefs are kept for 120 runs in the drop-down at the top of the site.
