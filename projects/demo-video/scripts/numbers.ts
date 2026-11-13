// Numbers spelled as read: normalizes digit tokens in narration into the
// words a voice actually says, so TTS audio and captions match. Idempotent —
// text the author already spelled out passes through untouched.

const ONES = [
  "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
  "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
  "sixteen", "seventeen", "eighteen", "nineteen",
];

const TENS = [
  "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
  "ninety",
];

function intToWords(n: number): string {
  if (n < 20) return ONES[n];
  if (n < 100) {
    const tens = Math.floor(n / 10);
    const ones = n % 10;
    return ones === 0 ? TENS[tens] : `${TENS[tens]}-${ONES[ones]}`;
  }
  if (n < 1000) {
    const hundreds = Math.floor(n / 100);
    const rest = n % 100;
    return rest === 0
      ? `${ONES[hundreds]} hundred`
      : `${ONES[hundreds]} hundred ${intToWords(rest)}`;
  }
  if (n < 1_000_000) {
    const thousands = Math.floor(n / 1000);
    const rest = n % 1000;
    return rest === 0
      ? `${intToWords(thousands)} thousand`
      : `${intToWords(thousands)} thousand ${intToWords(rest)}`;
  }
  const millions = Math.floor(n / 1_000_000);
  const rest = n % 1_000_000;
  return rest === 0
    ? `${intToWords(millions)} million`
    : `${intToWords(millions)} million ${intToWords(rest)}`;
}

function decimalToWords(value: string): string {
  const [whole, frac] = value.split(".");
  const wholeWords = intToWords(Number(whole.replace(/,/g, "")));
  const fracWords = frac!
    .split("")
    .map((d) => ONES[Number(d)])
    .join(" ");
  return `${wholeWords} point ${fracWords}`;
}

// One number token, optionally with a trailing "%".
function tokenToWords(token: string): string {
  const percent = token.endsWith("%");
  const digits = percent ? token.slice(0, -1) : token;
  const words = digits.includes(".")
    ? decimalToWords(digits)
    : intToWords(Number(digits.replace(/,/g, "")));
  return percent ? `${words} percent` : words;
}

// Matches integers and decimals (with optional thousands commas), optionally
// followed by %. Leaves words, dates written as words, and "$" alone untouched.
const NUMBER_TOKEN = /\d[\d,]*(?:\.\d+)?%?/g;

export function spellAsRead(text: string): string {
  return text.replace(NUMBER_TOKEN, (token) => tokenToWords(token));
}
