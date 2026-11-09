//! Shared top-first board encoding for the exported placement network.
//! Callers must convert their engine's board orientation before encoding.

pub const FEATURE_DIM: usize = 49;
const PIECES: [char; 7] = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];

pub fn encode(
    board: &[[u8; 10]],
    lines: usize,
    held: Option<char>,
    next: &[char],
    landing_height: f32,
) -> Result<[f32; FEATURE_DIM], &'static str> {
    if !(20..=80).contains(&board.len()) || lines > 4 || !landing_height.is_finite() {
        return Err("Invalid board height, clear count or landing height");
    }
    if held.is_some_and(|p| !PIECES.contains(&p)) || next.iter().any(|p| !PIECES.contains(p)) {
        return Err("Invalid hold or preview piece");
    }
    let mut occupied = [[false; 10]; 24];
    let count = board.len().min(24);
    for (destination, source) in occupied[24 - count..].iter_mut().zip(board[board.len() - count..].iter()) {
        for (cell, value) in destination.iter_mut().zip(source.iter()) { *cell = *value != 0; }
    }
    let mut heights = [0usize; 10];
    let mut covered = [0usize; 10];
    let (mut holes, mut covered_holes, mut row_transitions, mut col_transitions) = (0, 0, 0, 0);
    for (y, row) in occupied.iter().enumerate() {
        let mut previous = true;
        for (x, &cell) in row.iter().enumerate() {
            if cell {
                if heights[x] == 0 { heights[x] = 24 - y; }
                covered[x] += 1;
            } else if covered[x] > 0 { holes += 1; covered_holes += covered[x]; }
            row_transitions += usize::from(previous != cell);
            previous = cell;
        }
        row_transitions += usize::from(!previous);
    }
    for x in 0..10 {
        let mut previous = true;
        for row in &occupied {
            col_transitions += usize::from(previous != row[x]);
            previous = row[x];
        }
        col_transitions += usize::from(!previous);
    }
    let bumpiness: usize = heights.windows(2).map(|pair| pair[0].abs_diff(pair[1])).sum();
    let mut wells = 0;
    for x in 0..10 {
        let left = if x == 0 { 24 } else { heights[x - 1] };
        let right = if x == 9 { 24 } else { heights[x + 1] };
        wells += left.min(right).saturating_sub(heights[x]);
    }
    let mut features = [0.0; FEATURE_DIM];
    for x in 0..10 { features[x] = (heights[x] as f64 / 24.0) as f32; }
    let metrics = [
        heights.iter().sum::<usize>() as f64 / 240.0, holes as f64 / 240.0,
        bumpiness as f64 / 216.0, *heights.iter().max().unwrap() as f64 / 24.0,
        wells as f64 / 240.0, row_transitions as f64 / 264.0,
        col_transitions as f64 / 250.0, covered_holes as f64 / 2880.0,
        lines as f64 / 4.0, landing_height as f64 / 24.0,
    ];
    for (i, &value) in metrics.iter().enumerate() { features[10 + i] = value as f32; }
    features[20] = if held.is_none() { 1.0 } else { 0.0 };
    for (i, &piece) in PIECES.iter().enumerate() {
        features[21 + i] = if held == Some(piece) { 1.0 } else { 0.0 };
        for n in 0..3 { features[28 + 7 * n + i] = if next.get(n) == Some(&piece) { 1.0 } else { 0.0 }; }
    }
    Ok(features)
}
