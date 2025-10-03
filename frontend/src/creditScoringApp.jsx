// CreditScoringApp.jsx — 17 fields on the left, 16 on the right
// (Glassy selects/inputs, symmetric columns with a vertical divider)

import React, { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Smartphone,
  WalletMinimal,
  PiggyBank,
  Gauge,
  ChevronRight,
  Loader2,
  X,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

/* ---------- Small UI Primitives (glassy) ---------- */
const Label = ({ children }) => (
  <span className="text-xs md:text-sm font-medium text-slate-200/90 tracking-wide">
    {children}
  </span>
);

const Field = ({ label, children }) => (
  <label className="grid gap-1.5">
    <Label>{label}</Label>
    {children}
  </label>
);

const Input = ({ className = "", ...props }) => (
  <input
    className={`input-glass h-11 w-full rounded-xl px-3 py-2.5 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-300/50 ${className}`}
    {...props}
  />
);

/* High-contrast Select (needs .select-contrast in index.css) */
function Select({ className = "", children, ...props }) {
  return (
    <div className="relative">
      <select
        className={`input-glass select-contrast h-11 appearance-none w-full rounded-xl px-3 pr-9 focus:outline-none focus:ring-2 focus:ring-emerald-300/50 ${className}`}
        {...props}
      >
        {children}
      </select>
      <svg
        className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-emerald-200/90"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m6 9 6 6 6-6" />
      </svg>
    </div>
  );
}

/* ---------- Modal (Result Popup) ---------- */
function Modal({ open, onClose, children }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          aria-modal="true"
          role="dialog"
          className="fixed inset-0 z-50 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <div className="absolute inset-0 bg-black/55 backdrop-blur-sm" onClick={onClose} />
          <motion.div
            className="relative z-10 w-[92%] max-w-2xl rounded-2xl glass-card"
            initial={{ y: 28, scale: 0.98, opacity: 0 }}
            animate={{ y: 0, scale: 1, opacity: 1 }}
            exit={{ y: 16, scale: 0.98, opacity: 0 }}
            transition={{ type: "spring", stiffness: 120, damping: 16 }}
          >
            <div className="flex items-center justify-between px-5 py-4 border-b border-white/30">
              <div className="flex items-center gap-2 text-slate-50 font-semibold">
                <Gauge className="size-5 text-emerald-200" />
                Result
              </div>
              <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 text-slate-200">
                <X className="size-4" />
              </button>
            </div>
            <div className="p-5">{children}</div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ---------- Main App ---------- */
export default function CreditScoringApp() {
  const [form, setForm] = useState({
    Age: "",
    City: "",
    "Are you currently a student ?": "No",
    "What is your highest/current education level ?": "",
    "Are you currently employed ?": "Yes",
    "If Yes, what is your monthly income ?": "",
    "Do you use UPI ?": "",
    "If yes, how many UPI transactions do you make in a week?": 0,
    "What is your average monthly spending ?": "",
    "Do you pay rent ?": "No",
    "If yes, how much you pay monthly ?": "",
    "How do you pay your rent ?": "None",
    "Do you pay bills (electricity, mobile, internet) on time?": "Always",
    "How much do you spend on mobile recharges monthly?": "",
    "How many subscriptions do you pay for?": 0,
    "List your paid subscriptions": "",
    "How often are you unable to pay subscriptions on time?": "Never",
    "What phone type do you use?": "",
    "Phone brand and model ?": "",
    "How many hours/day do you use your phone?": 0,
    "Do you actively save a portion of your income?": "",
    "If yes, how much do you save per month": "",
    "If you save money, where do you keep your savings ?": "",
    "Do you use savings tracking applications ?": "",
    "Which financial apps do you use regularly?": "GPay, Groww",
    "Do you have any active EMIs or loans (student loan, mobile EMI, etc.)?": "",
    "If yes, total EMI amount paid per month?": "",
    "Have you ever used Buy Now Pay Later (BNPL) services (ZestMoney, LazyPay, Simpl)?": "",
    "Do you have a financial goal you're saving for?": "No",
    "If yes, what is the goal?": "",
    "How soon do you aim to achieve this goal (months)?": 0,
    "Do you have an emergency fund that can cover 3+ months of expenses?": "",
    "If faced with an emergency, how would you manage funds?": "",
    "Do you invest in crypto ?": "No",
    "Do you invest in any of the following ?": "Savings, Mutual Fund",
    "Do you participate in charity donations or volunteering?": "No",
    "How long have you been living at your current address (months) ?": 0,
    "How long have you been at your current job/college (months)?": 0,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);

  const update = (k, v) => setForm((prev) => ({ ...prev, [k]: v }));

  async function submit() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      setResult(data);
      setOpen(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const income = Number(form["If Yes, what is your monthly income ?"]) || 0;
  const spend = Number(form["What is your average monthly spending ?"]) || 0;
  const rent = Number(form["If yes, how much you pay monthly ?"]) || 0;
  const emi = Number(form["If yes, total EMI amount paid per month?"]) || 0;

  const burden = useMemo(() => ((rent + emi) / Math.max(income, 1)) * 100, [
    income,
    rent,
    emi,
  ]);

  return (
    <div className="min-h-screen bg-solid-aurora text-slate-100">
      {/* center the single glass panel */}
      <div className="min-h-screen flex items-center justify-center px-4 py-8">
        <div className="glass-card w-full max-w-6xl rounded-3xl p-6 md:p-8">
          {/* header */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 pb-4 border-b border-white/30">
            <div className="flex items-center gap-3">
              <div className="grid place-items-center size-12 rounded-2xl bg-white/10 border border-white/30">
                <Gauge className="size-6 text-emerald-200" />
              </div>
              <div>
                <h1 className="text-xl md:text-2xl font-semibold tracking-wide">
                  Alternate CIBIL Scoring
                </h1>
                <p className="text-slate-300/80 text-sm">
                  Enter details → get a CIBIL-like score with reasons
                </p>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 text-slate-200/85">
              <div className="flex items-center gap-2">
                <Smartphone className="size-4" /> UPI/wk:&nbsp;
                {form["If yes, how many UPI transactions do you make in a week?"]}
              </div>
              <div className="flex items-center gap-2">
                <WalletMinimal className="size-4" /> Spend: ₹{spend || 0}
              </div>
              <div className="flex items-center gap-2">
                <PiggyBank className="size-4" /> Income: ₹{income || 0}
              </div>
            </div>
          </div>

          {/* split form INSIDE single panel — symmetric columns with divider */}
          <div className="mt-6 grid lg:grid-cols-[1fr_auto_1fr] gap-6 lg:gap-10">
            {/* LEFT (17 fields) */}
            <div className="grid grid-cols-6 gap-3">
              <div className="col-span-3">
                <Field label="Age">
                  <Input type="number" value={form["Age"]} onChange={(e) => update("Age", e.target.value)} placeholder="20" />
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="City">
                  <Input value={form["City"]} onChange={(e) => update("City", e.target.value)} placeholder="Vellore" />
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Student?">
                  <Select value={form["Are you currently a student ?"]} onChange={(e) => update("Are you currently a student ?", e.target.value)}>
                    <option>Yes</option><option>No</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Education">
                  <Input value={form["What is your highest/current education level ?"]} onChange={(e) => update("What is your highest/current education level ?", e.target.value)} placeholder="Graduate" />
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Employed?">
                  <Select value={form["Are you currently employed ?"]} onChange={(e) => update("Are you currently employed ?", e.target.value)}>
                    <option>Yes</option><option>No</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Monthly Income (₹)">
                  <Input value={form["If Yes, what is your monthly income ?"]} onChange={(e) => update("If Yes, what is your monthly income ?", e.target.value)} placeholder="30000" />
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="UPI user?">
                  <Select value={form["Do you use UPI ?"]} onChange={(e) => update("Do you use UPI ?", e.target.value)}>
                    <option>Yes</option><option>No</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="UPI txns/week">
                  <Input type="number" value={form["If yes, how many UPI transactions do you make in a week?"]} onChange={(e) => update("If yes, how many UPI transactions do you make in a week?", Number(e.target.value))} />
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Monthly Spending (₹)">
                  <Input value={form["What is your average monthly spending ?"]} onChange={(e) => update("What is your average monthly spending ?", e.target.value)} />
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Pays Rent?">
                  <Select value={form["Do you pay rent ?"]} onChange={(e) => update("Do you pay rent ?", e.target.value)}>
                    <option>Yes</option><option>No</option>
                  </Select>
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Rent Amount (₹)">
                  <Input value={form["If yes, how much you pay monthly ?"]} onChange={(e) => update("If yes, how much you pay monthly ?", e.target.value)} />
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Rent Mode">
                  <Select value={form["How do you pay your rent ?"]} onChange={(e) => update("How do you pay your rent ?", e.target.value)}>
                    <option>Bank</option><option>UPI</option><option>Cash</option><option>None</option>
                  </Select>
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Bills on time">
                  <Select value={form["Do you pay bills (electricity, mobile, internet) on time?"]} onChange={(e) => update("Do you pay bills (electricity, mobile, internet) on time?", e.target.value)}>
                    <option>Always</option><option>Sometimes</option><option>Never</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Recharge spend (₹)">
                  <Input value={form["How much do you spend on mobile recharges monthly?"]} onChange={(e) => update("How much do you spend on mobile recharges monthly?", e.target.value)} />
                </Field>
              </div>

              {/* moved from right to left to make 17 total */}
              <div className="col-span-3">
                <Field label="Active EMIs?">
                  <Select value={form["Do you have any active EMIs or loans (student loan, mobile EMI, etc.)?"]} onChange={(e) => update("Do you have any active EMIs or loans (student loan, mobile EMI, etc.)?", e.target.value)}>
                    <option>No</option><option>Yes</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Total EMI / month (₹)">
                  <Input value={form["If yes, total EMI amount paid per month?"]} onChange={(e) => update("If yes, total EMI amount paid per month?", e.target.value)} />
                </Field>
              </div>
              <div className="col-span-6">
                <Field label="Used BNPL?">
                  <Select value={form["Have you ever used Buy Now Pay Later (BNPL) services (ZestMoney, LazyPay, Simpl)?"]} onChange={(e) => update("Have you ever used Buy Now Pay Later (BNPL) services (ZestMoney, LazyPay, Simpl)?", e.target.value)}>
                    <option>No</option><option>Yes</option>
                  </Select>
                </Field>
              </div>
            </div>

            {/* vertical divider */}
            <div className="hidden lg:block w-px bg-white/25 rounded-full" />

            {/* RIGHT (16 fields) */}
            <div className="grid grid-cols-6 gap-3">
              <div className="col-span-3">
                <Field label="Subscriptions count">
                  <Input type="number" value={form["How many subscriptions do you pay for?"]} onChange={(e) => update("How many subscriptions do you pay for?", Number(e.target.value))} />
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Sub payments lateness">
                  <Select value={form["How often are you unable to pay subscriptions on time?"]} onChange={(e) => update("How often are you unable to pay subscriptions on time?", e.target.value)}>
                    <option>Never</option><option>Sometimes</option><option>Always</option>
                  </Select>
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Phone type">
                  <Select value={form["What phone type do you use?"]} onChange={(e) => update("What phone type do you use?", e.target.value)}>
                    <option>Budget</option><option>Mid-Range</option><option>Flagship</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Phone hours/day">
                  <Input type="number" value={form["How many hours/day do you use your phone?"]} onChange={(e) => update("How many hours/day do you use your phone?", Number(e.target.value))} />
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Actively saving?">
                  <Select value={form["Do you actively save a portion of your income?"]} onChange={(e) => update("Do you actively save a portion of your income?", e.target.value)}>
                    <option>Yes</option><option>No</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Save per month (₹)">
                  <Input value={form["If yes, how much do you save per month"]} onChange={(e) => update("If yes, how much do you save per month", e.target.value)} />
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Savings app user?">
                  <Select value={form["Do you use savings tracking applications ?"]} onChange={(e) => update("Do you use savings tracking applications ?", e.target.value)}>
                    <option>No</option><option>Yes</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Financial apps (comma-separated)">
                  <Input value={form["Which financial apps do you use regularly?"]} onChange={(e) => update("Which financial apps do you use regularly?", e.target.value)} />
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Financial goal?">
                  <Select value={form["Do you have a financial goal you're saving for?"]} onChange={(e) => update("Do you have a financial goal you're saving for?", e.target.value)}>
                    <option>No</option><option>Yes</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Goal months">
                  <Input type="number" value={form["How soon do you aim to achieve this goal (months)?"]} onChange={(e) => update("How soon do you aim to achieve this goal (months)?", Number(e.target.value))} />
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Emergency fund (≥3 months)?">
                  <Select value={form["Do you have an emergency fund that can cover 3+ months of expenses?"]} onChange={(e) => update("Do you have an emergency fund that can cover 3+ months of expenses?", e.target.value)}>
                    <option>Yes</option><option>No</option>
                  </Select>
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Invest in crypto?">
                  <Select value={form["Do you invest in crypto ?"]} onChange={(e) => update("Do you invest in crypto ?", e.target.value)}>
                    <option>No</option><option>Yes</option>
                  </Select>
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Investments (text)">
                  <Input value={form["Do you invest in any of the following ?"]} onChange={(e) => update("Do you invest in any of the following ?", e.target.value)} />
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Charity / volunteering?">
                  <Select value={form["Do you participate in charity donations or volunteering?"]} onChange={(e) => update("Do you participate in charity donations or volunteering?", e.target.value)}>
                    <option>No</option><option>Yes</option>
                  </Select>
                </Field>
              </div>

              <div className="col-span-3">
                <Field label="Address months">
                  <Input value={form["How long have you been living at your current address (months) ?"]} onChange={(e) => update("How long have you been living at your current address (months) ?", e.target.value)} />
                </Field>
              </div>
              <div className="col-span-3">
                <Field label="Job/College months">
                  <Input value={form["How long have you been at your current job/college (months)?"]} onChange={(e) => update("How long have you been at your current job/college (months)?", e.target.value)} />
                </Field>
              </div>

              {/* Actions row spans full right side */}
              <div className="col-span-6 flex flex-wrap items-center gap-3 pt-2">
                <motion.button
                  onClick={submit}
                  disabled={loading}
                  whileTap={{ scale: 0.98 }}
                  className="btn-primary inline-flex items-center gap-2 rounded-xl px-5 py-2.5 shadow-lg disabled:opacity-60"
                >
                  {loading ? (
                    <>
                      <Loader2 className="size-4 animate-spin" /> Scoring…
                    </>
                  ) : (
                    <>
                      Get Score <ChevronRight className="size-4" />
                    </>
                  )}
                </motion.button>
                {error && <div className="text-rose-300/90 text-sm">{error}</div>}
                <div className="ml-auto text-xs text-slate-300/80">
                  Burden: {isFinite(burden) ? burden.toFixed(0) : 0}%
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Result Modal */}
      <Modal open={open} onClose={() => setOpen(false)}>
        {!result ? (
          <div className="text-slate-300/80 text-sm">
            Fill the form and click <span className="font-semibold">Get Score</span>.
          </div>
        ) : (
          <motion.div
            key="result"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="grid gap-4"
          >
            <div className="flex items-center gap-4">
              <div
                className="text-5xl md:text-6xl font-black tracking-tight text-white"
                style={{ textShadow: "0 0 30px rgba(94,234,212,.35)" }}
              >
                {result.cibil_like_score}
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wider text-slate-300">
                  CIBIL-like score
                </div>
                <div className="mt-1 text-sm md:text-base font-semibold">
                  {result.score_band}
                </div>
                <div className="text-xs text-slate-300">
                  Prob. good: {(result.prob_good * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="border-t border-white/30 pt-4">
              <div className="text-sm font-semibold mb-2">Top reasons</div>
              <ul className="list-disc pl-5 leading-relaxed text-slate-100">
                {result.top_reasons ? (
                  result.top_reasons.split(";").map((r, i) => <li key={i}>{r.trim()}</li>)
                ) : (
                  <li>No strong risk signals detected</li>
                )}
              </ul>
            </div>

            {(result.shap_top_positive || result.shap_top_negative) && (
              <div className="border-t border-white/30 pt-4 grid gap-2">
                {result.shap_top_positive && (
                  <div className="text-sm">
                    <span className="font-semibold">Positive:</span>{" "}
                    {result.shap_top_positive}
                  </div>
                )}
                {result.shap_top_negative && (
                  <div className="text-sm">
                    <span className="font-semibold">Negative:</span>{" "}
                    {result.shap_top_negative}
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </Modal>
    </div>
  );
}
