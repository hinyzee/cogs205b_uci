### Reflection 
I first wrote task.txt with class structure, method names, and 
constraints, but left the binomial formula and spike interval bounds 
vague to see if Gemma could derive them.

For test file safeguarding beyond the soft constraints, I used three layers. chmod 0o444 at startup. 
Snapshot the test file before the loop and restore it after every API call.
Whitelist the output path in write_files so only bayes_factor.py gets written; 
anything else is refused and logged.

On the first run attempt 1 returned a 500 (server error) from Gemini. 
Rerunning worked, so it was likely a server-side hiccup on Google's end 
rather than anything in the request. 

The returned code is not safe from code smell: evidence_slab returned 1/(n+1) 
as a closed-form shortcut instead of integrating (tests passed) 
But this can likely be solved by specifying in the prompt. 
The n=0 branch in bayes_factor is unnecessary because the math already returns 1.0 there. 
I would remove it and let evidence_spike / evidence_slab handle the case naturally.

