import { useState, useEffect } from "react";
import { useAuth } from "../hooks/useAuth";

/**
 * POPIA Privacy Consent popup component.
 *
 * Blocks access to the application until the user acknowledges
 * data storage practices and checks the consent checkbox.
 * Consent state is persisted in localStorage per user.
 *
 * Args:
 *     children (React.ReactNode): The app content to reveal after consent.
 *
 * Returns:
 *     JSX.Element: Either the consent modal overlay or the children.
 */
export const PopiaConsent = ({ children }: { children: React.ReactNode }) => {
    const { user } = useAuth();
    const [accepted, setAccepted] = useState(false);
    const [checked, setChecked] = useState(false);

    // Reason: Include user ID in key so each account gets their own consent prompt
    const consentKey = `popia_consent_accepted_${user?.id ?? "unknown"}`;

    useEffect(() => {
        const stored = localStorage.getItem(consentKey);
        if (stored === "true") {
            setAccepted(true);
        }
    }, [consentKey]);

    const handleAccept = () => {
        if (checked) {
            localStorage.setItem(consentKey, "true");
            setAccepted(true);
        }
    };

    if (accepted) {
        return <>{children}</>;
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
            <div className="mx-4 max-w-lg w-full rounded-2xl bg-[#1a1a2e] border border-[#cc8c0f]/30 shadow-2xl shadow-[#cc8c0f]/10 p-8">
                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                    <div className="h-10 w-10 rounded-full bg-[#cc8c0f]/20 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#cc8c0f]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                        </svg>
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-white">Privacy Notice</h2>
                        <p className="text-xs text-gray-400">POPIA Compliance</p>
                    </div>
                </div>

                {/* Policy content */}
                <div className="mb-6 space-y-4 text-sm text-gray-300 leading-relaxed">
                    <p>
                        Welcome to the <span className="text-[#cc8c0f] font-semibold">SWEAT AI Assistant</span>. In accordance with the
                        Protection of Personal Information Act (POPIA), we want to be transparent about the data we collect and store.
                    </p>

                    <div className="rounded-lg bg-white/5 p-4 space-y-2">
                        <p className="font-semibold text-white text-sm">By using this service, you acknowledge that we store:</p>
                        <ul className="list-disc list-inside space-y-1 text-gray-300">
                            <li>Your <span className="text-white font-medium">name</span> and <span className="text-white font-medium">email address</span> (via Google sign-in)</li>
                            <li>Your <span className="text-white font-medium">complete conversation history</span> with the AI assistant</li>
                        </ul>
                    </div>

                    <p>
                        This data is used solely to provide you with a personalised experience and is stored securely.
                        We do not share your personal information with third parties. You may request deletion of your data
                        at any time by contacting the event organisers.
                    </p>
                </div>

                {/* Checkbox */}
                <label className="flex items-start gap-3 mb-6 cursor-pointer group">
                    <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => setChecked(e.target.checked)}
                        className="mt-0.5 h-5 w-5 rounded border-gray-500 accent-[#cc8c0f] cursor-pointer"
                    />
                    <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
                        I have read and agree to the above privacy terms and consent to the storage of my personal information as described.
                    </span>
                </label>

                {/* Button */}
                <button
                    onClick={handleAccept}
                    disabled={!checked}
                    className={`w-full py-3 rounded-lg font-semibold text-white transition-all duration-200 ${checked
                        ? "bg-[#cc8c0f] hover:bg-[#b07a0d] cursor-pointer shadow-lg shadow-[#cc8c0f]/20"
                        : "bg-gray-600 cursor-not-allowed opacity-50"
                        }`}
                >
                    Continue to SWEAT AI
                </button>

                <p className="mt-4 text-center text-xs text-gray-500">
                    Protected under POPIA — Protection of Personal Information Act, 2013
                </p>
            </div>
        </div>
    );
};
