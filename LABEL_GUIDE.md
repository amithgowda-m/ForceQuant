# Labelling guide: Porter's forces in 10-K paragraphs

Label each paragraph with every force it clearly discusses as a threat or pressure on THIS company. Use `None` if it is boilerplate (legal, accounting, forward-looking statements) or off-topic. A paragraph can have 0 to 3 labels; label only what the paragraph is mainly about.

| Label | Use when the paragraph says... | Examples |
|---|---|---|
| Competitive_Rivalry | the company faces intense competition from existing rivals, price cuts, bundling, feature races, market share loss | "competitors offer similar products at lower prices", "intense competition" |
| Threat_of_New_Entry | new or emerging companies can enter easily, low barriers, startups, new technologies lowering entry costs, regulation changing entry | "new entrants", "low barriers to entry", "emerging competitors" |
| Threat_of_Substitutes | customers can use a different kind of product, open-source, in-house builds, cloud-provider managed services, alternative technologies | "customers may build in-house", "open source alternatives", "hyperscalers' managed services" |
| Buyer_Power | customers can negotiate prices or terms, switch easily, delay purchases, concentrate purchasing, demand discounts | "customers may demand price concessions", "low switching costs", "large customers" |
| Supplier_Power | the company depends on a few suppliers or partners for infrastructure, chips, hosting, licences, with cost or supply risk | "we rely on third-party data centers", "limited number of GPU suppliers" |
| Complementors | the company depends on third-party developers, integrators, marketplaces, app ecosystems, or partner products that add value | "our ecosystem of partners", "third-party apps on our platform", "marketplace" |

## Rules
1. A mention of the word is not enough. The paragraph must describe the pressure itself.
2. Competition from cloud providers can be Rivalry (they sell the same thing) or Substitutes (they offer a managed alternative). Use Substitutes when the paragraph is about customers choosing a different way to get the same outcome; use Rivalry when it is about direct competing products.
3. If unsure between a force and None, choose None.
4. Keep labels as a semicolon-separated string, e.g. `Competitive_Rivalry;Buyer_Power`, or `None`.
5. Do not use outside knowledge about the company, only the paragraph text.