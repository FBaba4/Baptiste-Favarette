// ── Abstract: Personnage ──────────────────────────────────────────────────────
class Personnage {
  constructor(nom, pointsDeVie, positionX, positionY) {
    if (new.target === Personnage) throw new Error("Personnage est abstraite");
    this.nom = nom;
    this.pointsDeVie = pointsDeVie;
    this.pointsDeVieMax = pointsDeVie;
    this.positionX = positionX;
    this.positionY = positionY;
  }
  estVivant() { return this.pointsDeVie > 0; }
  subirDegats(degats) { this.pointsDeVie = Math.max(0, this.pointsDeVie - degats); }
  soigner(soin) { this.pointsDeVie = Math.min(this.pointsDeVieMax, this.pointsDeVie + soin); }
}

// ── Abstract: Objet ───────────────────────────────────────────────────────────
class Objet {
  constructor(nom, description) {
    if (new.target === Objet) throw new Error("Objet est abstraite");
    this.nom = nom;
    this.description = description;
  }
}

class Arme extends Objet {
  constructor(nom, description, degats) {
    super(nom, description);
    this.degats = degats;
    this.type = "arme";
  }
}

class Consommable extends Objet {
  constructor(nom, description, soin) {
    super(nom, description);
    this.soin = soin;
    this.type = "consommable";
  }
}

// ── Inventaire ────────────────────────────────────────────────────────────────
class Inventaire {
  constructor() { this.contenu = []; }

  ajouterObjet(objet) {
    this.contenu.push(objet);
    return `${objet.nom} ajouté à l'inventaire.`;
  }

  retirerObjet(objet) {
    const idx = this.contenu.indexOf(objet);
    if (idx !== -1) { this.contenu.splice(idx, 1); return true; }
    return false;
  }

  getArmes() { return this.contenu.filter(o => o.type === "arme"); }
  getConsommables() { return this.contenu.filter(o => o.type === "consommable"); }
}

// ── Quete ─────────────────────────────────────────────────────────────────────
class Quete {
  constructor(description, recompense, condition) {
    this.description = description;
    this.estAccomplie = false;
    this.recompense = recompense;
    this.condition = condition;
  }

  validerObjectif(heros) {
    if (!this.estAccomplie && this.condition(heros)) {
      this.estAccomplie = true;
      return true;
    }
    return false;
  }
}

// ── Carte ─────────────────────────────────────────────────────────────────────
class Carte {
  constructor(nomLieu, largeur, hauteur) {
    this.nomLieu = nomLieu;
    this.largeur = largeur;
    this.hauteur = hauteur;
    this.pnjPresents = [];
    this.ennemis = [];
    this.objetsAuSol = [];
    this.grille = this._genererGrille();
  }

  _genererGrille() {
    const g = [];
    for (let y = 0; y < this.hauteur; y++) {
      g[y] = [];
      for (let x = 0; x < this.largeur; x++) {
        // Bordures = mur, reste = sol
        g[y][x] = (x === 0 || y === 0 || x === this.largeur - 1 || y === this.hauteur - 1) ? "mur" : "sol";
      }
    }
    return g;
  }

  estAccessible(x, y) {
    return x >= 0 && y >= 0 && x < this.largeur && y < this.hauteur && this.grille[y][x] !== "mur";
  }

  afficher() { return this.grille; }

  ajouterEnnemi(ennemi) { this.ennemis.push(ennemi); }
  ajouterObjetSol(objet, x, y) { this.objetsAuSol.push({ objet, x, y }); }

  getEnnemisEnPosition(x, y) {
    return this.ennemis.filter(e => e.positionX === x && e.positionY === y && e.estVivant());
  }

  getObjetEnPosition(x, y) {
    return this.objetsAuSol.find(o => o.x === x && o.y === y);
  }

  retirerObjetSol(objet) {
    const idx = this.objetsAuSol.indexOf(objet);
    if (idx !== -1) this.objetsAuSol.splice(idx, 1);
  }
}

// ── Ennemi ────────────────────────────────────────────────────────────────────
class Ennemi extends Personnage {
  constructor(nom, pointsDeVie, positionX, positionY, degatsInfliges, xpDonne = 20) {
    super(nom, pointsDeVie, positionX, positionY);
    this.degatsInfliges = degatsInfliges;
    this.xpDonne = xpDonne;
  }

  attaquer(cible) {
    const degats = Math.max(1, this.degatsInfliges - Math.floor(Math.random() * 3));
    cible.subirDegats(degats);
    return degats;
  }
}

class Boss extends Ennemi {
  constructor(nom, pointsDeVie, positionX, positionY, degatsInfliges) {
    super(nom, pointsDeVie, positionX, positionY, degatsInfliges, 100);
    this.estUnBoss = true;
  }

  attaquer(cible) {
    const degats = this.degatsInfliges + Math.floor(Math.random() * 5);
    cible.subirDegats(degats);
    return degats;
  }
}

// ── Heros ─────────────────────────────────────────────────────────────────────
class Heros extends Personnage {
  constructor(nom, pointsDeVie, positionX, positionY, classe) {
    super(nom, pointsDeVie, positionX, positionY);
    this.classe = classe;
    this.sacADos = new Inventaire();
    this.queteEnCours = null;
    this.quetesAccomplies = [];
    this.xp = 0;
    this.niveau = 1;
    this.armureEquipee = null;
    this.armeEquipee = null;
    this.enDefense = false;
    this.ennemisVaincus = 0;
    this.bossVaincus = 0;
  }

  seDeplacer(direction, carte) {
    let nx = this.positionX;
    let ny = this.positionY;
    if (direction === "haut")    ny--;
    if (direction === "bas")     ny++;
    if (direction === "gauche")  nx--;
    if (direction === "droite")  nx++;

    if (carte.estAccessible(nx, ny)) {
      this.positionX = nx;
      this.positionY = ny;
      return true;
    }
    return false;
  }

  attaquer(cible) {
    const base = this.armeEquipee ? this.armeEquipee.degats : 5;
    const degats = base + Math.floor(Math.random() * 4);
    cible.subirDegats(degats);
    return degats;
  }

  defendre() {
    this.enDefense = true;
    return Math.floor(this.pointsDeVieMax * 0.1);
  }

  utiliserObjet(objet) {
    if (objet.type === "consommable") {
      this.soigner(objet.soin);
      this.sacADos.retirerObjet(objet);
      return `${this.nom} utilise ${objet.nom} et récupère ${objet.soin} PV.`;
    }
    if (objet.type === "arme") {
      this.armeEquipee = objet;
      return `${this.nom} équipe ${objet.nom} (${objet.degats} dégâts).`;
    }
    return "Impossible d'utiliser cet objet.";
  }

  gagnerXP(xp) {
    this.xp += xp;
    const xpNecessaire = this.niveau * 50;
    if (this.xp >= xpNecessaire) {
      this.xp -= xpNecessaire;
      this.niveau++;
      this.pointsDeVieMax += 10;
      this.pointsDeVie = Math.min(this.pointsDeVie + 10, this.pointsDeVieMax);
      return true; // level up
    }
    return false;
  }

  accepterQuete(quete) {
    this.queteEnCours = quete;
  }
}

// ── Jeu ───────────────────────────────────────────────────────────────────────
class Jeu {
  constructor() {
    this.enPause = false;
    this.heros = null;
    this.carte = null;
    this.quetes = [];
    this.log = [];
    this.combatEnCours = false;
    this.ennemisVaincusCarte = 0;
    this._initialiserQuetes();
  }

  _initialiserQuetes() {
    this.quetes = [
      new Quete(
        "Vaincre 3 ennemis",
        new Consommable("Grande Potion", "Restaure 40 PV", 40),
        (h) => h.ennemisVaincus >= 3
      ),
      new Quete(
        "Vaincre le Boss de la Forêt",
        new Arme("Épée Légendaire", "Lame enchantée", 25),
        (h) => h.bossVaincus >= 1
      ),
      new Quete(
        "Collecter 2 objets",
        new Consommable("Élixir", "Restaure 60 PV", 60),
        (h) => h.sacADos.contenu.length >= 2
      ),
    ];
  }

  mettreEnPause() {
    this.enPause = !this.enPause;
    return this.enPause;
  }

  sauvegarderProgression() {
    const data = {
      heros: {
        nom: this.heros.nom,
        classe: this.heros.classe,
        pointsDeVie: this.heros.pointsDeVie,
        pointsDeVieMax: this.heros.pointsDeVieMax,
        positionX: this.heros.positionX,
        positionY: this.heros.positionY,
        xp: this.heros.xp,
        niveau: this.heros.niveau,
        ennemisVaincus: this.heros.ennemisVaincus,
        bossVaincus: this.heros.bossVaincus,
        inventaire: this.heros.sacADos.contenu.map(o => ({
          type: o.type, nom: o.nom, description: o.description,
          valeur: o.type === "arme" ? o.degats : o.soin
        })),
        armeEquipee: this.heros.armeEquipee ? {
          nom: this.heros.armeEquipee.nom,
          description: this.heros.armeEquipee.description,
          degats: this.heros.armeEquipee.degats
        } : null,
      },
      quetes: this.quetes.map(q => ({ description: q.description, estAccomplie: q.estAccomplie })),
      nomLieu: this.carte.nomLieu,
    };
    localStorage.setItem("rpg_save", JSON.stringify(data));
    return "Progression sauvegardée !";
  }

  chargerProgression() {
    const raw = localStorage.getItem("rpg_save");
    if (!raw) return null;
    return JSON.parse(raw);
  }

  lancerCombat(heros, ennemi) {
    this.combatEnCours = true;
    this.combatant = { heros, ennemi, tour: "heros" };
    return this.combatant;
  }

  ajouterLog(msg) {
    this.log.unshift(msg);
    if (this.log.length > 50) this.log.pop();
  }

  verifierQuetes() {
    const accomplies = [];
    for (const quete of this.quetes) {
      if (!quete.estAccomplie && quete.validerObjectif(this.heros)) {
        accomplies.push(quete);
      }
    }
    return accomplies;
  }
}
