// ── Canvas Renderer ───────────────────────────────────────────────────────────
const TILE = 48;

// Positions des arbres sur la carte (inspirées du campus)
const ARBRES = [
  {x:2,y:2},{x:13,y:2},{x:2,y:9},{x:13,y:9},
  {x:6,y:5},{x:9,y:5},{x:5,y:8},{x:10,y:3},
];

// Positions herbe/pelouse
const HERBE = [
  {x:1,y:1},{x:2,y:1},{x:1,y:2},
  {x:12,y:1},{x:13,y:1},{x:13,y:2},
  {x:1,y:9},{x:1,y:10},{x:2,y:10},
  {x:12,y:10},{x:13,y:10},{x:13,y:9},
  {x:6,y:4},{x:7,y:4},{x:8,y:4},{x:9,y:4},
  {x:5,y:7},{x:6,y:7},{x:9,y:8},{x:10,y:8},
];

let animFrame = 0;
let animId = null;

// ── Dessin d'une tuile sol (dalle béton campus) ───────────────────────────────
function drawSol(ctx, px, py, variant) {
  const shade = variant % 3;
  ctx.fillStyle = shade === 0 ? '#c2bab0' : shade === 1 ? '#bbb3a8' : '#c8c0b5';
  ctx.fillRect(px, py, TILE, TILE);

  // Joints entre dalles
  ctx.strokeStyle = 'rgba(0,0,0,0.08)';
  ctx.lineWidth = 1;
  ctx.strokeRect(px + 0.5, py + 0.5, TILE - 1, TILE - 1);

  // Légère texture
  ctx.fillStyle = 'rgba(255,255,255,0.04)';
  ctx.fillRect(px + 2, py + 2, TILE - 4, TILE / 2 - 2);
}

// ── Dessin d'un mur / bâtiment (façade moderne) ───────────────────────────────
function drawMur(ctx, px, py) {
  // Façade principale
  const grad = ctx.createLinearGradient(px, py, px + TILE, py + TILE);
  grad.addColorStop(0, '#ddd8ce');
  grad.addColorStop(1, '#c8c2b8');
  ctx.fillStyle = grad;
  ctx.fillRect(px, py, TILE, TILE);

  // Fenêtre (style bâtiment vitré campus)
  const winX = px + 8, winY = py + 8, winW = TILE - 16, winH = TILE - 18;
  ctx.fillStyle = '#a8cce0';
  ctx.fillRect(winX, winY, winW, winH);

  // Reflet vitre
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.fillRect(winX + 2, winY + 2, winW / 2 - 2, winH / 2 - 2);

  // Cadre fenêtre
  ctx.strokeStyle = '#b0a898';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(winX, winY, winW, winH);
  // Séparation vitre
  ctx.beginPath();
  ctx.moveTo(winX + winW / 2, winY);
  ctx.lineTo(winX + winW / 2, winY + winH);
  ctx.moveTo(winX, winY + winH / 2);
  ctx.lineTo(winX + winW, winY + winH / 2);
  ctx.stroke();

  // Contour bâtiment
  ctx.strokeStyle = 'rgba(0,0,0,0.15)';
  ctx.lineWidth = 1;
  ctx.strokeRect(px, py, TILE, TILE);
}

// ── Dessin de la pelouse ──────────────────────────────────────────────────────
function drawHerbe(ctx, px, py) {
  ctx.fillStyle = '#4e7c42';
  ctx.fillRect(px, py, TILE, TILE);

  // Texture herbe (brins)
  ctx.fillStyle = '#3d6433';
  for (let i = 0; i < 6; i++) {
    const bx = px + 5 + i * 7;
    const by = py + 10 + (i % 2) * 8;
    ctx.fillRect(bx, by, 2, 12);
    ctx.fillRect(bx + 2, by + 4, 2, 8);
  }
  ctx.fillStyle = '#5a8a50';
  ctx.fillRect(px + 2, py + 2, TILE - 4, 8);
}

// ── Dessin d'un arbre ─────────────────────────────────────────────────────────
function drawArbre(ctx, px, py, t) {
  const cx = px + TILE / 2;
  const cy = py + TILE / 2;
  const bob = Math.sin(t * 0.03) * 1.5;

  // Ombre
  ctx.fillStyle = 'rgba(0,0,0,0.18)';
  ctx.beginPath();
  ctx.ellipse(cx + 4, cy + 16, 12, 5, 0, 0, Math.PI * 2);
  ctx.fill();

  // Tronc
  ctx.fillStyle = '#7a5230';
  ctx.fillRect(cx - 4, cy + 4, 8, 16);

  // Feuillage (3 cercles superposés)
  ctx.fillStyle = '#2a5c1a';
  ctx.beginPath();
  ctx.arc(cx, cy - 6 + bob, 14, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#3a7028';
  ctx.beginPath();
  ctx.arc(cx - 6, cy - 2 + bob, 10, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#3a7028';
  ctx.beginPath();
  ctx.arc(cx + 6, cy - 2 + bob, 10, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#4a8035';
  ctx.beginPath();
  ctx.arc(cx, cy - 10 + bob, 10, 0, Math.PI * 2);
  ctx.fill();

  // Reflet lumière
  ctx.fillStyle = 'rgba(255,255,255,0.12)';
  ctx.beginPath();
  ctx.arc(cx - 4, cy - 12 + bob, 5, 0, Math.PI * 2);
  ctx.fill();
}

// ── Dessin du héros ───────────────────────────────────────────────────────────
function drawHeros(ctx, heros, t) {
  const px = heros.positionX * TILE;
  const py = heros.positionY * TILE;
  const cx = px + TILE / 2;
  const cy = py + TILE / 2;
  const pulse = Math.sin(t * 0.08) * 3;

  const couleurs = {
    guerrier: { corps: '#2a6ab0', accent: '#4a9ae0', glow: 'rgba(70,130,220,0.4)' },
    archer:   { corps: '#2a8a40', accent: '#4aba60', glow: 'rgba(70,180,90,0.4)' },
    mage:     { corps: '#7a30a0', accent: '#b060e0', glow: 'rgba(160,80,220,0.4)' },
  };
  const c = couleurs[heros.classe] || couleurs.guerrier;

  // Halo animé
  ctx.fillStyle = c.glow;
  ctx.beginPath();
  ctx.arc(cx, cy, 18 + pulse, 0, Math.PI * 2);
  ctx.fill();

  // Ombre
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  ctx.beginPath();
  ctx.ellipse(cx + 3, cy + 14, 11, 4, 0, 0, Math.PI * 2);
  ctx.fill();

  // Corps
  ctx.fillStyle = c.corps;
  ctx.beginPath();
  ctx.arc(cx, cy + 2, 13, 0, Math.PI * 2);
  ctx.fill();

  // Tête
  ctx.fillStyle = '#f0d0a0';
  ctx.beginPath();
  ctx.arc(cx, cy - 7, 8, 0, Math.PI * 2);
  ctx.fill();

  // Détail classe
  ctx.fillStyle = c.accent;
  if (heros.classe === 'guerrier') {
    // Bouclier
    ctx.beginPath();
    ctx.roundRect(cx + 6, cy - 4, 8, 12, 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillRect(cx + 9, cy - 1, 2, 6);
    ctx.fillRect(cx + 7, cy + 1, 6, 2);
  } else if (heros.classe === 'archer') {
    // Arc
    ctx.strokeStyle = c.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx + 10, cy, 8, -Math.PI * 0.6, Math.PI * 0.6);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx + 10, cy - 6);
    ctx.lineTo(cx + 10, cy + 6);
    ctx.strokeStyle = '#d4a820';
    ctx.stroke();
  } else {
    // Baguette magique
    ctx.lineWidth = 3;
    ctx.strokeStyle = c.accent;
    ctx.beginPath();
    ctx.moveTo(cx + 5, cy + 8);
    ctx.lineTo(cx + 14, cy - 4);
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(cx + 14, cy - 5, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // Contour
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy + 2, 13, 0, Math.PI * 2);
  ctx.stroke();

  // Indicateur nom
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(cx - 20, cy - 30, 40, 14);
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 9px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(heros.nom.slice(0, 6), cx, cy - 19);
}

// ── Dessin d'un ennemi ────────────────────────────────────────────────────────
function drawEnnemi(ctx, ennemi, t) {
  const px = ennemi.positionX * TILE;
  const py = ennemi.positionY * TILE;
  const cx = px + TILE / 2;
  const cy = py + TILE / 2;
  const pulse = Math.sin(t * 0.06 + px) * 2;

  if (ennemi.estUnBoss) {
    // Dragon (boss)
    ctx.fillStyle = `rgba(200,40,20,${0.3 + Math.sin(t * 0.05) * 0.15})`;
    ctx.beginPath();
    ctx.arc(cx, cy, 22 + pulse, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#8b1a10';
    ctx.beginPath();
    ctx.arc(cx, cy, 17, 0, Math.PI * 2);
    ctx.fill();

    // Yeux du dragon
    ctx.fillStyle = '#ff6600';
    ctx.beginPath(); ctx.arc(cx - 6, cy - 4, 4, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(cx + 6, cy - 4, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#ffff00';
    ctx.beginPath(); ctx.arc(cx - 6, cy - 4, 2, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(cx + 6, cy - 4, 2, 0, Math.PI * 2); ctx.fill();

    // Flamme
    ctx.fillStyle = `rgba(255,120,0,${0.7 + Math.sin(t * 0.1) * 0.3})`;
    ctx.beginPath();
    ctx.moveTo(cx, cy + 8);
    ctx.quadraticCurveTo(cx + 8, cy + 20 + pulse, cx, cy + 28);
    ctx.quadraticCurveTo(cx - 8, cy + 20 + pulse, cx, cy + 8);
    ctx.fill();

    // Label BOSS
    ctx.fillStyle = 'rgba(150,0,0,0.8)';
    ctx.fillRect(cx - 22, cy - 32, 44, 14);
    ctx.fillStyle = '#ffdd00';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('BOSS: ' + ennemi.nom.slice(0, 8), cx, cy - 21);
  } else {
    // Ennemi normal
    ctx.fillStyle = 'rgba(180,30,30,0.3)';
    ctx.beginPath();
    ctx.arc(cx, cy, 16 + pulse, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#5a1010';
    ctx.beginPath();
    ctx.arc(cx, cy, 11, 0, Math.PI * 2);
    ctx.fill();

    // Yeux rouges
    ctx.fillStyle = '#ff2020';
    ctx.beginPath(); ctx.arc(cx - 4, cy - 2, 3, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(cx + 4, cy - 2, 3, 0, Math.PI * 2); ctx.fill();

    // Bouche
    ctx.strokeStyle = '#ff4040';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy + 4, 4, 0.2, Math.PI - 0.2);
    ctx.stroke();

    // Nom
    ctx.fillStyle = 'rgba(80,0,0,0.75)';
    ctx.fillRect(cx - 18, cy - 26, 36, 12);
    ctx.fillStyle = '#ffaaaa';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(ennemi.nom.slice(0, 8), cx, cy - 17);
  }

  // Barre de PV
  const barW = TILE - 6;
  const pct = ennemi.pointsDeVie / ennemi.pointsDeVieMax;
  ctx.fillStyle = '#300';
  ctx.fillRect(px + 3, py + TILE - 8, barW, 5);
  ctx.fillStyle = pct > 0.5 ? '#20c050' : pct > 0.25 ? '#f0a000' : '#e02020';
  ctx.fillRect(px + 3, py + TILE - 8, barW * pct, 5);
}

// ── Dessin d'un objet au sol ──────────────────────────────────────────────────
function drawObjetSol(ctx, objetSol, t) {
  const px = objetSol.x * TILE;
  const py = objetSol.y * TILE;
  const cx = px + TILE / 2;
  const cy = py + TILE / 2;
  const bob = Math.sin(t * 0.07 + px) * 3;

  const isArme = objetSol.objet.type === 'arme';

  // Lueur
  ctx.fillStyle = isArme ? 'rgba(220,180,30,0.35)' : 'rgba(30,180,120,0.35)';
  ctx.beginPath();
  ctx.arc(cx, cy, 15, 0, Math.PI * 2);
  ctx.fill();

  // Base de l'objet
  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.beginPath();
  ctx.ellipse(cx + 2, cy + 14, 8, 3, 0, 0, Math.PI * 2);
  ctx.fill();

  if (isArme) {
    // Épée/arme
    ctx.save();
    ctx.translate(cx, cy - 2 + bob);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = '#d4a820';
    ctx.fillRect(-3, -14, 6, 28);
    ctx.fillStyle = '#b8922a';
    ctx.fillRect(-8, -2, 16, 5);
    ctx.fillStyle = '#e8e8e8';
    ctx.fillRect(-2, -14, 4, 24);
    ctx.restore();
  } else {
    // Potion
    ctx.save();
    ctx.translate(cx, cy - 2 + bob);
    ctx.fillStyle = '#c030d0';
    ctx.fillRect(-5, -8, 10, 14);
    ctx.fillStyle = '#a020b0';
    ctx.fillRect(-3, 2, 6, 4);
    ctx.fillStyle = '#888';
    ctx.fillRect(-3, -12, 6, 5);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fillRect(-3, -7, 3, 7);
    ctx.restore();
  }
}

// ── Rendu complet de la grille ────────────────────────────────────────────────
function dessinerCarte(canvas, jeu, t) {
  const ctx = canvas.getContext('2d');
  const carte = jeu.carte;
  const grille = carte.afficher();

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Sol de base (fond)
  ctx.fillStyle = '#a8a090';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Tuiles
  for (let y = 0; y < carte.hauteur; y++) {
    for (let x = 0; x < carte.largeur; x++) {
      const px = x * TILE, py = y * TILE;
      const type = grille[y][x];
      const estHerbe = HERBE.some(h => h.x === x && h.y === y);
      const estArbre = ARBRES.some(a => a.x === x && a.y === y);

      if (type === 'mur') {
        drawMur(ctx, px, py);
      } else if (estHerbe || estArbre) {
        drawHerbe(ctx, px, py);
      } else {
        drawSol(ctx, px, py, x + y);
      }
    }
  }

  // Arbres (par-dessus le sol)
  ARBRES.forEach(a => {
    if (jeu.carte.grille[a.y] && jeu.carte.grille[a.y][a.x] !== 'mur') {
      drawArbre(ctx, a.x * TILE, a.y * TILE, t);
    }
  });

  // Objets au sol
  carte.objetsAuSol.forEach(o => drawObjetSol(ctx, o, t));

  // Ennemis vivants
  carte.ennemis.filter(e => e.estVivant()).forEach(e => drawEnnemi(ctx, e, t));

  // Héros
  drawHeros(ctx, jeu.heros, t);

  // Vignette (assombrir les bords)
  const vign = ctx.createRadialGradient(
    canvas.width / 2, canvas.height / 2, canvas.width * 0.3,
    canvas.width / 2, canvas.height / 2, canvas.width * 0.75
  );
  vign.addColorStop(0, 'rgba(0,0,0,0)');
  vign.addColorStop(1, 'rgba(0,0,0,0.35)');
  ctx.fillStyle = vign;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// ── Légende de la carte ───────────────────────────────────────────────────────
function dessinerLegende(ctx, x, y) {
  const items = [
    { col: '#4e7c42', label: 'Pelouse' },
    { col: '#5a1010', label: 'Ennemi' },
    { col: '#8b1a10', label: 'Boss' },
    { col: '#d4a820', label: 'Arme' },
    { col: '#c030d0', label: 'Potion' },
    { col: '#2a6ab0', label: 'Héros' },
  ];
  ctx.font = '11px sans-serif';
  items.forEach((it, i) => {
    ctx.fillStyle = it.col;
    ctx.fillRect(x, y + i * 18, 12, 12);
    ctx.fillStyle = '#ddd';
    ctx.fillText(it.label, x + 16, y + i * 18 + 10);
  });
}

// ── Lancement de la boucle d'animation ───────────────────────────────────────
function demarrerAnimation(canvasId, jeu) {
  if (animId) cancelAnimationFrame(animId);
  animFrame = 0;

  function loop() {
    const canvas = document.getElementById(canvasId);
    if (!canvas) { animId = null; return; }
    dessinerCarte(canvas, jeu, animFrame);
    animFrame++;
    animId = requestAnimationFrame(loop);
  }
  loop();
}

function stopperAnimation() {
  if (animId) { cancelAnimationFrame(animId); animId = null; }
}
