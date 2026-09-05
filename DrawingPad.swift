//
//  DrawingPad.swift
//  neural net v1
//
//  Created by Oliver Homer on 01/09/2026.
//

import SwiftUI
import NeuralApp

struct Stroke {
    var points: [CGPoint]
}


struct DrawingCanvas: View {
    let strokes: [Stroke]
    let currentStroke: Stroke
    let sourceSize: CGSize

    var body: some View {
        Canvas { context, size in
            // Scale from the source coordinate space (e.g., 400x400) into current canvas size (e.g., 28x28)
            let sx = size.width / max(sourceSize.width, 1)
            let sy = size.height / max(sourceSize.height, 1)
            context.scaleBy(x: sx, y: sy)

            for stroke in strokes {
                draw(stroke, in: &context)
            }

            draw(currentStroke, in: &context)
        }
        .background(Color.black)
    }
}


struct DrawingPad: View {
    
    @State private var strokes: [Stroke] = []
    @State private var currentStroke = Stroke(points: [])
    private let padSize = CGSize(width: 400, height: 400)
    @StateObject private var engineBox: EngineBox

    public init(engine: AppEngine) {
        _engineBox = StateObject(wrappedValue: EngineBox(engine: engine))
    }
    
    var body: some View {
        DrawingCanvas(strokes: strokes, currentStroke: currentStroke, sourceSize: padSize)
            .frame(width: padSize.width, height: padSize.height)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        currentStroke.points.append(value.location)
                    }
                    .onEnded { value in
                        currentStroke.points.append(value.location)
                        strokes.append(currentStroke)
                        currentStroke = Stroke(points: [])
                    }
            )
            .padding()
        
        HStack {
            Button("Rasterise") {
                let pixels = rasterisewithImageRenderer()
                let nonZero = pixels.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }
                print("pixels.count=\(pixels.count), nonZero=\(nonZero)")

                pixels.withUnsafeBufferPointer { buffer in
                    if let base = buffer.baseAddress {
                        engineBox.sendRasterData(base, pixels.count)
                    }
                }
            }
            Button("Clear") {
                strokes.removeAll()
                currentStroke = Stroke(points: [])
            }
        }
        .padding()
    }
    
    func rasterisewithImageRenderer() -> [Float] {
        let drawingView = DrawingCanvas(
            strokes: strokes,
            currentStroke: currentStroke,
            sourceSize: padSize
        )
        .frame(width: 28, height: 28)
        
        let renderer = ImageRenderer(content: drawingView)
        renderer.proposedSize = ProposedViewSize(width: 28, height: 28)
        renderer.scale = 1.0
        
        guard let cgImage = renderer.cgImage else {
            return []
        }
        
        return grayscalePixels(from: cgImage)
    }
    
    func grayscalePixels(from cgImage: CGImage) -> [Float] {
        let width = cgImage.width
        let height = cgImage.height
        
        var pixels = [UInt8](repeating: 0, count: width * height)
        
        guard let context = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            return []
        }
    
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return pixels.map { Float($0) / 255.0 }
    }
    
    
}

func draw(_ stroke: Stroke, in context: inout GraphicsContext) {
    guard stroke.points.count > 1 else {return}
    
    var path = Path()
    path.move(to: stroke.points[0])
    
    for point in stroke.points.dropFirst() {
        path.addLine(to: point)
    }
    
    context.stroke(
        path,
        with: .color(.white),
        style: StrokeStyle(lineWidth: 12, lineCap: .round, lineJoin: .round)
    )
}

