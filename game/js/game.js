// ── Initialisation ────────────────────────────────────────────────────────────
const jeu = new Jeu();
let etat = "selection"; // selection | exploration | combat | inventaire | carte | quetes | pause | fin

const PERSONNAGES_DISPONIBLES = [
  { nom: "Guerrier", pv: 100, classe: "guerrier",
    armeDepart: new Arme("Épée courte", "Solide et fiable", 10),
    desc: "Robuste et puissant. PV élevés, dégâts moyens." },
  { nom: "Archer",   pv: 75,  classe: "archer",
    armeDepart: new Arme("Arc en bois", "Précision à distance", 14),
    desc: "Agile et précis. Moins de PV mais plus de dégâts." },
  { nom: "Mage",     pv: 60,  classe: "mage",
    armeDepart: new Arme("Bâton magique", "Canalise la magie", 18),
    desc: "Fragile mais dévastateur. Dégâts très élevés." },
];

// ── Construction de la carte ───────────────────────────────────────────────────
function creerCarte(nomLieu) {
  const carte = new Carte(nomLieu, 15, 12);

  // Murs supplémentaires aléatoires (obstacles)
  const obstacles = [
    [3,3],[3,4],[7,2],[7,3],[11,5],[11,6],[4,8],[5,8],[9,9],[10,9],
    [6,5],[2,7],[12,3],[8,7]
  ];
  obstacles.forEach(([x, y]) => { carte.grille[y][x] = "mur"; });

  // Ennemis
  carte.ajouterEnnemi(new Ennemi("Gobelin",    30, 4, 2, 8,  15));
  carte.ajouterEnnemi(new Ennemi("Squelette",  40, 9, 4, 10, 20));
  carte.ajouterEnnemi(new Ennemi("Loup",       35, 5, 7, 9,  18));
  carte.ajouterEnnemi(new Ennemi("Orc",        50, 12,8, 12, 25));
  carte.ajouterEnnemi(new Boss("Dragon Ancien",120,11,2, 20, 100));

  // Objets au sol
  carte.ajouterObjetSol(new Consommable("Potion", "Restaure 25 PV", 25), 6, 3);
  carte.ajouterObjetSol(new Arme("Hache", "Arme lourde", 15), 2, 5);
  carte.ajouterObjetSol(new Consommable("Herbes médicinales", "Restaure 15 PV", 15), 10, 6);
  carte.ajouterObjetSol(new Consommable("Grande Potion", "Restaure 40 PV", 40), 13, 10);

  return carte;
}

// ── Rendu ─────────────────────────────────────────────────────────────────────
function render() {
  const app = document.getElementById("app");
  if (!app) return;

  switch (etat) {
    case "selection": app.innerHTML = renderSelection(); break;
    case "exploration": app.innerHTML = renderExploration(); break;
    case "combat": app.innerHTML = renderCombat(); break;
    case "inventaire": app.innerHTML = renderInventaire(); break;
    case "carte": app.innerHTML = renderCarte(); break;
    case "quetes": app.innerHTML = renderQuetes(); break;
    case "pause": app.innerHTML = renderPause(); break;
    case "fin": app.innerHTML = renderFin(); break;
  }
  bindEvents();
}

// ── Écrans ────────────────────────────────────────────────────────────────────
function renderSelection() {
  const hasSave = !!localStorage.getItem("rpg_save");
  return `
    <div class="screen selection-screen">
      <h1 class="titre">⚔ QUEST RPG ⚔</h1>
      <p class="subtitle">Choisissez votre personnage</p>
      <div class="personnages-grid">
        ${PERSONNAGES_DISPONIBLES.map((p, i) => `
          <div class="personnage-card" data-index="${i}">
            <div class="perso-icon">${p.classe === "guerrier" ? "🛡️" : p.classe === "archer" ? "🏹" : "🔮"}</div>
            <h3>${p.nom}</h3>
            <p class="perso-pv">❤️ ${p.pv} PV</p>
            <p class="perso-arme">⚔️ ${p.armeDepart.degats} dégâts</p>
            <p class="perso-desc">${p.desc}</p>
            <button class="btn btn-choisir" data-index="${i}">Choisir</button>
          </div>
        `).join("")}
      </div>
      ${hasSave ? `<button class="btn btn-charger" id="btn-charger">📂 Charger la sauvegarde</button>` : ""}
    </div>`;
}

function renderExploration() {
  const h = jeu.heros;
  const grille = jeu.carte.afficher();
  const cellSize = 38;

  let gridHTML = '<div class="grille">';
  for (let y = 0; y < jeu.carte.hauteur; y++) {
    gridHTML += '<div class="ligne">';
    for (let x = 0; x < jeu.carte.largeur; x++) {
      const estHeros = (x === h.positionX && y === h.positionY);
      const ennemis = jeu.carte.getEnnemisEnPosition(x, y);
      const objetSol = jeu.carte.getObjetEnPosition(x, y);
      const type = grille[y][x];

      let contenu = "";
      let cls = `cellule cellule-${type}`;

      if (estHeros) { contenu = h.classe === "guerrier" ? "🧙" : h.classe === "archer" ? "🏹" : "🔮"; cls += " cellule-heros"; }
      else if (ennemis.length > 0) { contenu = ennemis[0].estUnBoss ? "🐉" : "👹"; cls += " cellule-ennemi"; }
      else if (objetSol) { contenu = objetSol.objet.type === "arme" ? "⚔️" : "🧪"; cls += " cellule-objet"; }
      else if (type === "mur") { contenu = ""; }

      gridHTML += `<div class="${cls}" style="width:${cellSize}px;height:${cellSize}px">${contenu}</div>`;
    }
    gridHTML += '</div>';
  }
  gridHTML += '</div>';

  const xpNecessaire = h.niveau * 50;
  const pctPV = Math.round((h.pointsDeVie / h.pointsDeVieMax) * 100);
  const pctXP = Math.round((h.xp / xpNecessaire) * 100);

  return `
    <div class="screen exploration-screen">
      <div class="hud">
        <div class="hud-left">
          <span class="hud-nom">${h.classe === "guerrier" ? "🛡️" : h.classe === "archer" ? "🏹" : "🔮"} ${h.nom}</span>
          <span class="hud-niveau">Niv. ${h.niveau}</span>
          <div class="barre-container">
            <div class="barre barre-pv" style="width:${pctPV}%"></div>
            <span class="barre-label">❤️ ${h.pointsDeVie}/${h.pointsDeVieMax}</span>
          </div>
          <div class="barre-container">
            <div class="barre barre-xp" style="width:${pctXP}%"></div>
            <span class="barre-label">⭐ ${h.xp}/${xpNecessaire} XP</span>
          </div>
        </div>
        <div class="hud-right">
          <span class="lieu">📍 ${jeu.carte.nomLieu}</span>
          ${h.armeEquipee ? `<span class="arme-equipee">⚔️ ${h.armeEquipee.nom}</span>` : ""}
        </div>
      </div>

      <div class="jeu-zone">
        ${gridHTML}
        <div class="panneau-lateral">
          <div class="log-zone">
            <h4>Journal</h4>
            ${jeu.log.slice(0, 8).map(l => `<p class="log-msg">${l}</p>`).join("")}
          </div>
          <div class="controles-hint">
            <p>⬆⬇⬅➡ Se déplacer</p>
            <p>[I] Inventaire</p>
            <p>[M] Carte/Quêtes</p>
            <p>[P] Pause</p>
          </div>
        </div>
      </div>
    </div>`;
}

function renderCombat() {
  const { heros: h, ennemi: e } = jeu.combatant;
  const pctHerosPV = Math.round((h.pointsDeVie / h.pointsDeVieMax) * 100);
  const pctEnnemPV = Math.round((e.pointsDeVie / e.pointsDeVieMax) * 100);

  return `
    <div class="screen combat-screen">
      <h2 class="combat-titre">⚔️ Combat !</h2>
      <div class="combat-arena">
        <div class="combattant combattant-heros">
          <div class="combattant-icon">${h.classe === "guerrier" ? "🛡️" : h.classe === "archer" ? "🏹" : "🔮"}</div>
          <h3>${h.nom}</h3>
          <div class="barre-container">
            <div class="barre barre-pv" style="width:${pctHerosPV}%"></div>
            <span class="barre-label">❤️ ${h.pointsDeVie}/${h.pointsDeVieMax}</span>
          </div>
          ${h.armeEquipee ? `<p class="arme-info">⚔️ ${h.armeEquipee.nom}</p>` : ""}
          ${h.enDefense ? `<p class="defense-actif">🛡️ En défense !</p>` : ""}
        </div>
        <div class="vs">VS</div>
        <div class="combattant combattant-ennemi">
          <div class="combattant-icon">${e.estUnBoss ? "🐉" : "👹"}</div>
          <h3>${e.nom}${e.estUnBoss ? " (BOSS)" : ""}</h3>
          <div class="barre-container">
            <div class="barre barre-ennemi" style="width:${pctEnnemPV}%"></div>
            <span class="barre-label">❤️ ${e.pointsDeVie}/${e.pointsDeVieMax}</span>
          </div>
          <p class="ennemi-info">⚡ ${e.degatsInfliges} dégâts</p>
        </div>
      </div>

      <div class="combat-log">
        ${jeu.log.slice(0, 5).map(l => `<p>${l}</p>`).join("")}
      </div>

      <div class="combat-actions">
        <button class="btn btn-attaquer" id="btn-attaquer">⚔️ Attaquer</button>
        <button class="btn btn-defendre" id="btn-defendre">🛡️ Se défendre</button>
        <button class="btn btn-objet-combat" id="btn-objet-combat">🎒 Utiliser objet</button>
        <button class="btn btn-fuir" id="btn-fuir">💨 Fuir</button>
      </div>

      ${renderMiniInventaireCombat()}
    </div>`;
}

function renderMiniInventaireCombat() {
  const consommables = jeu.heros.sacADos.getConsommables();
  if (consommables.length === 0) return "";
  return `
    <div class="mini-inventaire" id="mini-inv" style="display:none">
      <h4>Choisir un objet :</h4>
      ${consommables.map((o, i) => `
        <button class="btn btn-mini-objet" data-idx="${i}">${o.nom} (+${o.soin} PV)</button>
      `).join("")}
      <button class="btn btn-fermer-inv" id="btn-fermer-inv">✕ Fermer</button>
    </div>`;
}

function renderInventaire() {
  const inv = jeu.heros.sacADos;
  return `
    <div class="screen inventaire-screen">
      <h2>🎒 Inventaire de ${jeu.heros.nom}</h2>
      <div class="inventaire-grille">
        ${inv.contenu.length === 0 ? '<p class="vide">Inventaire vide</p>' :
          inv.contenu.map((o, i) => `
            <div class="item-card">
              <span class="item-icon">${o.type === "arme" ? "⚔️" : "🧪"}</span>
              <span class="item-nom">${o.nom}</span>
              <span class="item-desc">${o.description}</span>
              <span class="item-val">${o.type === "arme" ? `${o.degats} dégâts` : `+${o.soin} PV`}</span>
              <button class="btn btn-utiliser-inv" data-idx="${i}">
                ${o.type === "arme" ? "Équiper" : "Utiliser"}
              </button>
            </div>
          `).join("")}
      </div>
      <div class="arme-equipee-info">
        ${jeu.heros.armeEquipee ? `⚔️ Arme équipée : <strong>${jeu.heros.armeEquipee.nom}</strong> (${jeu.heros.armeEquipee.degats} dégâts)` : "Aucune arme équipée"}
      </div>
      <button class="btn btn-retour" id="btn-retour">↩ Retour</button>
    </div>`;
}

function renderCarte() {
  const ennemisRestants = jeu.carte.ennemis.filter(e => e.estVivant()).length;
  return `
    <div class="screen carte-screen">
      <h2>🗺️ ${jeu.carte.nomLieu}</h2>
      <div class="carte-info">
        <p>📍 Position : (${jeu.heros.positionX}, ${jeu.heros.positionY})</p>
        <p>👹 Ennemis restants : ${ennemisRestants}</p>
        <p>💎 Objets au sol : ${jeu.carte.objetsAuSol.length}</p>
      </div>
      <h3>📜 Quêtes</h3>
      <div class="quetes-liste">
        ${jeu.quetes.map(q => `
          <div class="quete-item ${q.estAccomplie ? "accomplie" : ""}">
            <span>${q.estAccomplie ? "✅" : "🔲"} ${q.description}</span>
            ${!q.estAccomplie ? "" : `<span class="recompense-label">(Récompense récupérée)</span>`}
          </div>
        `).join("")}
      </div>
      <div class="quete-active">
        ${jeu.heros.queteEnCours && !jeu.heros.queteEnCours.estAccomplie
          ? `<p>🎯 Quête active : <strong>${jeu.heros.queteEnCours.description}</strong></p>`
          : `<p>Aucune quête active.</p>`}
        ${jeu.quetes.filter(q => !q.estAccomplie && q !== jeu.heros.queteEnCours).map((q, i) => `
          <button class="btn btn-accepter-quete" data-qidx="${jeu.quetes.indexOf(q)}">🎯 Accepter : ${q.description}</button>
        `).join("")}
      </div>
      <button class="btn btn-retour" id="btn-retour">↩ Retour</button>
    </div>`;
}

function renderPause() {
  return `
    <div class="screen pause-screen">
      <h2>⏸ Pause</h2>
      <div class="pause-stats">
        <p>👤 ${jeu.heros.nom} — Niveau ${jeu.heros.niveau}</p>
        <p>❤️ ${jeu.heros.pointsDeVie}/${jeu.heros.pointsDeVieMax} PV</p>
        <p>⭐ ${jeu.heros.xp} XP</p>
        <p>👹 Ennemis vaincus : ${jeu.heros.ennemisVaincus}</p>
      </div>
      <div class="pause-actions">
        <button class="btn" id="btn-reprendre">▶️ Reprendre</button>
        <button class="btn" id="btn-sauvegarder">💾 Sauvegarder</button>
        <button class="btn" id="btn-inventaire-pause">🎒 Inventaire</button>
        <button class="btn btn-danger" id="btn-menu-principal">🏠 Menu principal</button>
      </div>
      <p id="msg-sauvegarde" class="msg-save"></p>
    </div>`;
}

function renderFin() {
  const victoire = jeu.heros && jeu.heros.estVivant();
  return `
    <div class="screen fin-screen">
      <h1>${victoire ? "🏆 Victoire !" : "💀 Défaite !"}</h1>
      <p>${victoire
        ? `${jeu.heros.nom} a triomphé et est devenu une légende !`
        : "Votre héros est tombé au combat..."}</p>
      <div class="fin-stats">
        <p>Niveau atteint : ${jeu.heros?.niveau || 1}</p>
        <p>Ennemis vaincus : ${jeu.heros?.ennemisVaincus || 0}</p>
        <p>Quêtes accomplies : ${jeu.quetes.filter(q => q.estAccomplie).length}</p>
      </div>
      <button class="btn" id="btn-rejouer">🔄 Rejouer</button>
    </div>`;
}

// ── Événements ─────────────────────────────────────────────────────────────────
function bindEvents() {
  // Sélection personnage
  document.querySelectorAll(".btn-choisir").forEach(btn => {
    btn.addEventListener("click", () => choisirPersonnage(parseInt(btn.dataset.index)));
  });

  document.getElementById("btn-charger")?.addEventListener("click", chargerJeu);

  // Combat
  document.getElementById("btn-attaquer")?.addEventListener("click", actionAttaquer);
  document.getElementById("btn-defendre")?.addEventListener("click", actionDefendre);
  document.getElementById("btn-objet-combat")?.addEventListener("click", () => {
    const mini = document.getElementById("mini-inv");
    if (mini) mini.style.display = mini.style.display === "none" ? "block" : "none";
  });
  document.getElementById("btn-fermer-inv")?.addEventListener("click", () => {
    document.getElementById("mini-inv").style.display = "none";
  });
  document.querySelectorAll(".btn-mini-objet").forEach(btn => {
    btn.addEventListener("click", () => {
      const consommables = jeu.heros.sacADos.getConsommables();
      const objet = consommables[parseInt(btn.dataset.idx)];
      if (objet) {
        const msg = jeu.heros.utiliserObjet(objet);
        jeu.ajouterLog(msg);
        render();
      }
    });
  });
  document.getElementById("btn-fuir")?.addEventListener("click", fuirCombat);

  // Inventaire
  document.querySelectorAll(".btn-utiliser-inv").forEach(btn => {
    btn.addEventListener("click", () => {
      const objet = jeu.heros.sacADos.contenu[parseInt(btn.dataset.idx)];
      if (objet) {
        const msg = jeu.heros.utiliserObjet(objet);
        jeu.ajouterLog(msg);
        render();
      }
    });
  });

  // Quêtes
  document.querySelectorAll(".btn-accepter-quete").forEach(btn => {
    btn.addEventListener("click", () => {
      const quete = jeu.quetes[parseInt(btn.dataset.qidx)];
      if (quete) {
        jeu.heros.accepterQuete(quete);
        jeu.ajouterLog(`Quête acceptée : ${quete.description}`);
        render();
      }
    });
  });

  // Navigation
  document.getElementById("btn-retour")?.addEventListener("click", () => { etat = "exploration"; render(); });
  document.getElementById("btn-reprendre")?.addEventListener("click", () => { etat = "exploration"; render(); });
  document.getElementById("btn-sauvegarder")?.addEventListener("click", () => {
    const msg = jeu.sauvegarderProgression();
    const el = document.getElementById("msg-sauvegarde");
    if (el) { el.textContent = msg; el.style.opacity = "1"; setTimeout(() => { el.style.opacity = "0"; }, 2000); }
  });
  document.getElementById("btn-inventaire-pause")?.addEventListener("click", () => { etat = "inventaire"; render(); });
  document.getElementById("btn-menu-principal")?.addEventListener("click", () => {
    localStorage.removeItem("rpg_save"); etat = "selection"; render();
  });
  document.getElementById("btn-rejouer")?.addEventListener("click", () => {
    localStorage.removeItem("rpg_save"); location.reload();
  });
}

// ── Clavier ───────────────────────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  if (etat !== "exploration") return;
  const touches = { ArrowUp:"haut", ArrowDown:"bas", ArrowLeft:"gauche", ArrowRight:"droite",
                    z:"haut", s:"bas", q:"gauche", d:"droite" };
  if (touches[e.key]) { e.preventDefault(); deplacer(touches[e.key]); return; }
  if (e.key === "i" || e.key === "I") { etat = "inventaire"; render(); return; }
  if (e.key === "m" || e.key === "M") { etat = "carte"; render(); return; }
  if (e.key === "p" || e.key === "P") { etat = "pause"; render(); return; }
});

// ── Logique ───────────────────────────────────────────────────────────────────
function choisirPersonnage(index) {
  const perso = PERSONNAGES_DISPONIBLES[index];
  jeu.heros = new Heros(perso.nom, perso.pv, 1, 1, perso.classe);
  jeu.heros.sacADos.ajouterObjet(perso.armeDepart);
  jeu.heros.armeEquipee = perso.armeDepart;
  jeu.carte = creerCarte("Forêt des Ombres");

  jeu.quetes[0].condition = (h) => h.ennemisVaincus >= 3;
  jeu.heros.accepterQuete(jeu.quetes[0]);

  jeu.ajouterLog(`${perso.nom} entre dans la Forêt des Ombres...`);
  jeu.ajouterLog("Utilisez les flèches pour vous déplacer.");
  etat = "exploration";
  render();
}

function chargerJeu() {
  const data = jeu.chargerProgression();
  if (!data) return;

  jeu.heros = new Heros(data.heros.nom, data.heros.pointsDeVieMax, data.heros.positionX, data.heros.positionY, data.heros.classe);
  jeu.heros.pointsDeVie = data.heros.pointsDeVie;
  jeu.heros.xp = data.heros.xp;
  jeu.heros.niveau = data.heros.niveau;
  jeu.heros.ennemisVaincus = data.heros.ennemisVaincus;
  jeu.heros.bossVaincus = data.heros.bossVaincus;

  data.heros.inventaire.forEach(o => {
    const obj = o.type === "arme" ? new Arme(o.nom, o.description, o.valeur) : new Consommable(o.nom, o.description, o.valeur);
    jeu.heros.sacADos.ajouterObjet(obj);
  });
  if (data.heros.armeEquipee) {
    jeu.heros.armeEquipee = new Arme(data.heros.armeEquipee.nom, data.heros.armeEquipee.description, data.heros.armeEquipee.degats);
  }

  jeu.quetes.forEach((q, i) => { if (data.quetes[i]) q.estAccomplie = data.quetes[i].estAccomplie; });
  jeu.carte = creerCarte(data.nomLieu || "Forêt des Ombres");

  jeu.ajouterLog("Partie chargée !");
  etat = "exploration";
  render();
}

function deplacer(direction) {
  const ok = jeu.heros.seDeplacer(direction, jeu.carte);
  if (!ok) return;

  // Objet au sol
  const objetSol = jeu.carte.getObjetEnPosition(jeu.heros.positionX, jeu.heros.positionY);
  if (objetSol) {
    const msg = jeu.heros.sacADos.ajouterObjet(objetSol.objet);
    jeu.carte.retirerObjetSol(objetSol);
    jeu.ajouterLog(`✨ ${msg}`);

    // Vérifier quêtes
    const accomplies = jeu.verifierQuetes();
    accomplies.forEach(q => {
      if (q.recompense) {
        jeu.heros.sacADos.ajouterObjet(q.recompense);
        jeu.ajouterLog(`🏆 Quête accomplie : "${q.description}" ! Récompense : ${q.recompense.nom}`);
      }
    });
  }

  // Ennemi sur la case
  const ennemis = jeu.carte.getEnnemisEnPosition(jeu.heros.positionX, jeu.heros.positionY);
  if (ennemis.length > 0) {
    jeu.lancerCombat(jeu.heros, ennemis[0]);
    jeu.ajouterLog(`⚔️ Combat contre ${ennemis[0].nom} !`);
    jeu.heros.enDefense = false;
    etat = "combat";
  }

  render();
}

function actionAttaquer() {
  const { heros: h, ennemi: e } = jeu.combatant;
  h.enDefense = false;

  const degatsHeros = h.attaquer(e);
  jeu.ajouterLog(`⚔️ ${h.nom} inflige ${degatsHeros} dégâts à ${e.nom}.`);

  if (!e.estVivant()) {
    jeu.ajouterLog(`💀 ${e.nom} est vaincu !`);
    h.ennemisVaincus++;
    if (e.estUnBoss) h.bossVaincus++;

    const levelUp = h.gagnerXP(e.xpDonne);
    jeu.ajouterLog(`⭐ +${e.xpDonne} XP gagnés !`);
    if (levelUp) jeu.ajouterLog(`🆙 ${h.nom} passe au niveau ${h.niveau} !`);

    // Loot aléatoire
    if (Math.random() < 0.4) {
      const loot = new Consommable("Potion de soin", "Trouvée sur l'ennemi", 20);
      h.sacADos.ajouterObjet(loot);
      jeu.ajouterLog(`💊 Loot : Potion de soin !`);
    }

    const accomplies = jeu.verifierQuetes();
    accomplies.forEach(q => {
      if (q.recompense) {
        h.sacADos.ajouterObjet(q.recompense);
        jeu.ajouterLog(`🏆 Quête : "${q.description}" ! Récompense : ${q.recompense.nom}`);
      }
    });

    jeu.combatant = null;
    jeu.combatEnCours = false;

    // Victoire si boss vaincu
    if (e.estUnBoss) { etat = "fin"; render(); return; }
    etat = "exploration";
    render();
    return;
  }

  // Tour ennemi
  tourEnnemi();
}

function actionDefendre() {
  const { heros: h } = jeu.combatant;
  const bonus = h.defendre();
  h.soigner(bonus);
  jeu.ajouterLog(`🛡️ ${h.nom} se défend et récupère ${bonus} PV.`);
  tourEnnemi();
}

function tourEnnemi() {
  const { heros: h, ennemi: e } = jeu.combatant;
  let degatsEnnemi = e.attaquer(h);

  if (h.enDefense) {
    degatsEnnemi = Math.floor(degatsEnnemi * 0.5);
    h.subirDegats(degatsEnnemi);
    jeu.ajouterLog(`👹 ${e.nom} attaque mais la défense réduit à ${degatsEnnemi} dégâts !`);
  } else {
    jeu.ajouterLog(`👹 ${e.nom} inflige ${degatsEnnemi} dégâts à ${h.nom}.`);
  }
  h.enDefense = false;

  if (!h.estVivant()) {
    jeu.ajouterLog("💀 Vous êtes mort...");
    jeu.combatant = null;
    etat = "fin";
  }
  render();
}

function fuirCombat() {
  jeu.ajouterLog("💨 Vous fuyez le combat !");
  jeu.heros.positionX = 1;
  jeu.heros.positionY = 1;
  jeu.combatant = null;
  jeu.combatEnCours = false;
  etat = "exploration";
  render();
}

// ── Démarrage ─────────────────────────────────────────────────────────────────
render();
